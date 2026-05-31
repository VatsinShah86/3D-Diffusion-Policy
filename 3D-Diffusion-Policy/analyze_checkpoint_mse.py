"""
Post-hoc per-dimension MSE analysis across training checkpoints.

Loads every epoch_N.ckpt in one or more run directories, runs the full
DDIM inference pipeline on the validation set, and reports MSE split by:
  pos_mse     — position  (action dims 0:3, metres²)
  rot_mse     — rotation  (action dims 3:6, rad²)
  gripper_mse — gripper   (action dim  6,   unitless²  [0-1 range])

Outputs per run dir:
  <run_dir>/analysis/checkpoint_mse.csv
  <run_dir>/analysis/checkpoint_mse.png

Usage:
    uv run python analyze_checkpoint_mse.py <run_dir> [<run_dir2> ...] [--max-batches N] [--device cuda:N]

Example (both abs_action models):
    uv run python analyze_checkpoint_mse.py \\
        data/outputs/real_cloth_manip_abs_action-dp3-ego_cloth_60eps_abs_action_seed0 \\
        data/outputs/real_cloth_manip_ext_seg_abs_action-dp3-ext_cloth_60eps_abs_action_seed0
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path

import dill
import hydra.utils
import numpy as np
import torch
from torch.utils.data import DataLoader

DP3_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(DP3_ROOT))
os.chdir(str(DP3_ROOT))

from omegaconf import OmegaConf
OmegaConf.register_new_resolver("eval", eval, replace=True)

from diffusion_policy_3d.common.pytorch_util import dict_apply


# ── Checkpoint discovery ──────────────────────────────────────────────────────

def find_epoch_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    """Return [(epoch_num, path), ...] sorted by epoch for all epoch_N.ckpt files."""
    ckpts = []
    for p in (run_dir / "checkpoints").glob("epoch_*.ckpt"):
        m = re.fullmatch(r"epoch_(\d+)\.ckpt", p.name)
        if m:
            ckpts.append((int(m.group(1)), p))
    return sorted(ckpts)


# ── Dataset / dataloader ──────────────────────────────────────────────────────

def build_val_dataloader(cfg, batch_size: int = 64) -> DataLoader:
    """Instantiate the validation dataset from the checkpoint config."""
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    val_dataset = dataset.get_validation_dataset()
    return DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        persistent_workers=False,
    )


# ── Per-checkpoint evaluation ────────────────────────────────────────────────

def eval_checkpoint(
    ckpt_path: Path,
    val_loader: DataLoader,
    max_batches: int,
    device: str,
) -> dict[str, float]:
    """
    Load a checkpoint, run inference on val_loader, return per-dim MSE.

    Returns {'pos_mse': float, 'rot_mse': float, 'gripper_mse': float}.
    """
    payload = torch.load(ckpt_path, pickle_module=dill, map_location="cpu")
    cfg     = payload["cfg"]

    # Build policy and load EMA weights only (skip optimizer / scheduler)
    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(payload["state_dicts"]["ema_model"])

    # Normalizer must be set before inference
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    policy.set_normalizer(dataset.get_normalizer())

    policy.eval().to(device)
    policy.reset()

    GRIP_THRESHOLD = 0.2   # matches policy_runtime.py GRIPPER_THRESHOLD

    pos_mse_acc = rot_mse_acc = grip_mse_acc = 0.0
    grip_close_steps = grip_total_steps = 0
    grip_bce_acc = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= max_batches:
                break

            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
            gt    = batch["action"]                                        # (B, horizon, 7)
            pred  = policy.predict_action(batch["obs"])["action_pred"]     # (B, horizon, 7)

            pos_mse_acc  += torch.nn.functional.mse_loss(pred[..., 0:3], gt[..., 0:3]).item()
            rot_mse_acc  += torch.nn.functional.mse_loss(pred[..., 3:6], gt[..., 3:6]).item()
            grip_mse_acc += torch.nn.functional.mse_loss(pred[..., 6:7], gt[..., 6:7]).item()

            # Fraction of predicted steps classified as CLOSE
            g_pred = pred[..., 6]                                          # (B, horizon)
            grip_close_steps += int((g_pred < GRIP_THRESHOLD).sum().item())
            grip_total_steps += int(g_pred.numel())

            # BCE: how well does the prediction distinguish open vs close?
            # Target is binary: gt > threshold → open (1), else close (0)
            g_gt_bin = (gt[..., 6] > GRIP_THRESHOLD).float()
            grip_bce_acc += torch.nn.functional.binary_cross_entropy(
                g_pred.clamp(1e-6, 1 - 1e-6), g_gt_bin
            ).item()

            n_batches += 1

    del policy, dataset
    torch.cuda.empty_cache()

    if n_batches == 0:
        nan = float("nan")
        return {"pos_mse": nan, "rot_mse": nan, "gripper_mse": nan,
                "gripper_close_frac": nan, "gripper_bce": nan}

    return {
        "pos_mse":           pos_mse_acc  / n_batches,
        "rot_mse":           rot_mse_acc  / n_batches,
        "gripper_mse":       grip_mse_acc / n_batches,
        "gripper_close_frac": grip_close_steps / grip_total_steps,
        "gripper_bce":       grip_bce_acc / n_batches,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(epochs: list[int], results: list[dict], run_dir: Path, run_name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos_mse    = [r["pos_mse"]            for r in results]
    rot_mse    = [r["rot_mse"]            for r in results]
    grip_mse   = [r["gripper_mse"]        for r in results]
    close_frac = [r["gripper_close_frac"] for r in results]
    grip_bce   = [r["gripper_bce"]        for r in results]

    # GT close fraction (constant — dataset property, not checkpoint-dependent)
    # Compute once from the first result's val data; approximate via training label stats.
    # We annotate the plot with gt_close_frac if available, else skip.

    fig, axes = plt.subplots(1, 5, figsize=(24, 5))
    fig.suptitle(f"{run_name}\nPer-dimension metrics vs training epoch", fontsize=11)

    def _panel(ax, y, title, color, ylabel, hline=None, hline_label=None):
        ax.plot(epochs, y, color=color, linewidth=1.2, marker=".", markersize=4)
        if hline is not None:
            ax.axhline(hline, color="red", ls="--", lw=0.9, label=hline_label or "")
            ax.legend(fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)

    _panel(axes[0], pos_mse,    "Position MSE",             "#4c78a8", "MSE  (m²)")
    _panel(axes[1], rot_mse,    "Rotation MSE",             "#f28e2b", "MSE  (rad²)")
    _panel(axes[2], grip_mse,   "Gripper MSE",              "#59a14f", "MSE  (unitless²)")
    _panel(axes[3], close_frac, "Predicted CLOSE fraction", "#e45756", "Fraction of steps",
           hline=0.34, hline_label="GT close frac ≈ 0.34")
    _panel(axes[4], grip_bce,   "Gripper BCE",              "#b279a2", "BCE loss")

    plt.tight_layout()

    out_dir = run_dir / "analysis"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "checkpoint_mse.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  [plot] saved → {out_path}")
    plt.close(fig)


# ── CSV ───────────────────────────────────────────────────────────────────────

def save_csv(epochs: list[int], results: list[dict], run_dir: Path) -> None:
    out_dir = run_dir / "analysis"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "checkpoint_mse.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "pos_mse", "rot_mse", "gripper_mse",
                                               "gripper_close_frac", "gripper_bce"])
        writer.writeheader()
        for ep, r in zip(epochs, results):
            writer.writerow({"epoch": ep, **r})
    print(f"  [csv]  saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def analyze_run(run_dir: Path, max_batches: int, device: str) -> None:
    ckpts = find_epoch_checkpoints(run_dir)
    if not ckpts:
        print(f"[skip] no epoch_N.ckpt files found in {run_dir}/checkpoints/")
        return

    print(f"\n{'='*70}")
    print(f"Run : {run_dir.name}")
    print(f"     {len(ckpts)} checkpoints  |  max_batches={max_batches}  |  device={device}")
    print(f"{'='*70}")

    # Build val_loader once from the first checkpoint's config
    first_payload = torch.load(ckpts[0][1], pickle_module=dill, map_location="cpu")
    val_loader    = build_val_dataloader(first_payload["cfg"], batch_size=64)
    del first_payload
    total_val_batches = min(max_batches, len(val_loader))
    print(f"  Validation batches per checkpoint: {total_val_batches} of {len(val_loader)}")

    epochs  = []
    results = []
    for i, (epoch, ckpt_path) in enumerate(ckpts):
        print(f"  [{i+1:2d}/{len(ckpts)}] epoch {epoch:5d}  ({ckpt_path.name})", end="", flush=True)
        r = eval_checkpoint(ckpt_path, val_loader, max_batches, device)
        print(f"  pos={r['pos_mse']:.5f}  rot={r['rot_mse']:.5f}  "
              f"grip_mse={r['gripper_mse']:.5f}  "
              f"close_frac={r['gripper_close_frac']:.4f}  bce={r['gripper_bce']:.4f}")
        epochs.append(epoch)
        results.append(r)

    save_csv(epochs, results, run_dir)
    plot_results(epochs, results, run_dir, run_dir.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", type=Path,
                        help="One or more training run directories (contain checkpoints/)")
    parser.add_argument("--max-batches", type=int, default=8,
                        help="Validation batches to use per checkpoint (default: 8)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Torch device (default: cuda if available)")
    args = parser.parse_args()

    for run_dir in args.run_dirs:
        run_dir = run_dir if run_dir.is_absolute() else DP3_ROOT / run_dir
        analyze_run(run_dir.resolve(), args.max_batches, args.device)

    print("\nAll done.")


if __name__ == "__main__":
    main()
