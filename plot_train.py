#!/usr/bin/env python3
import argparse
import glob
import json
import math
import os
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _candidate_roots(root: str) -> List[str]:
    # If user passed an absolute path, use it as-is.
    if os.path.isabs(root):
        return [root]
    here = os.path.dirname(os.path.abspath(__file__))
    return [
        root,
        os.path.join(here, root),
        os.path.join(here, "3D-Diffusion-Policy", root),
    ]


def _find_history_files(root: str) -> List[str]:
    files: List[str] = []
    for base in _candidate_roots(root):
        patterns = [
            os.path.join(base, "**", "wandb", "offline-run-*", "files", "wandb-history.jsonl"),
            os.path.join(base, "**", "wandb", "run-*", "files", "wandb-history.jsonl"),
        ]
        for pattern in patterns:
            files.extend(glob.glob(pattern, recursive=True))
    files = [f for f in files if os.path.isfile(f)]
    return files


def _latest_file(files: List[str]) -> str:
    if not files:
        raise FileNotFoundError("No wandb-history.jsonl files found.")
    return max(files, key=lambda p: os.path.getmtime(p))


def _find_wandb_run_files(root: str) -> List[str]:
    files: List[str] = []
    for base in _candidate_roots(root):
        patterns = [
            os.path.join(base, "**", "wandb", "offline-run-*", "run-*.wandb"),
            os.path.join(base, "**", "wandb", "run-*", "run-*.wandb"),
        ]
        for pattern in patterns:
            files.extend(glob.glob(pattern, recursive=True))
    files = [f for f in files if os.path.isfile(f)]
    return files


def _load_history(path: str) -> List[dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and not math.isnan(x)


def _parse_value(value_json: str):
    try:
        return json.loads(value_json)
    except Exception:
        # try to parse as a bare number
        try:
            return float(value_json)
        except Exception:
            return value_json


def _load_history_from_wandb(run_path: str) -> List[dict]:
    try:
        from wandb.sdk.internal.datastore import DataStore
        from wandb.proto import wandb_internal_pb2
    except Exception as e:
        raise RuntimeError("wandb is required to parse .wandb files") from e

    store = DataStore()
    store.open_for_scan(run_path)

    rows: List[dict] = []
    while True:
        data = store.scan_data()
        if data is None:
            break
        rec = wandb_internal_pb2.Record()
        try:
            rec.ParseFromString(data)
        except Exception:
            continue
        if not rec.HasField("history"):
            continue
        h = rec.history
        row: Dict[str, float] = {}
        step_val = None
        for item in h.item:
            key = item.key
            if not key and item.nested_key:
                key = "/".join(item.nested_key)
            if not key:
                continue
            val = _parse_value(item.value_json)
            if key == "_step":
                if _is_number(val):
                    step_val = int(val)
                continue
            if _is_number(val):
                row[key] = float(val)
        if step_val is not None:
            row["_step"] = step_val
        if row:
            rows.append(row)
    return rows


def _collect_series(rows: List[dict]) -> Dict[str, List[Tuple[int, float]]]:
    series: Dict[str, List[Tuple[int, float]]] = {}
    for idx, row in enumerate(rows):
        step = row.get("_step", idx)
        for k, v in row.items():
            if k == "_step":
                continue
            if _is_number(v):
                series.setdefault(k, []).append((step, float(v)))
    return series


def _sanitize(name: str) -> str:
    safe = name.replace("/", "_").replace(" ", "_")
    safe = safe.replace(":", "_").replace("=", "_")
    return safe


def _plot_all(series: Dict[str, List[Tuple[int, float]]], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for key, points in series.items():
        points = sorted(points, key=lambda p: p[0])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        if not xs:
            continue
        plt.figure(figsize=(8, 4.5))
        plt.plot(xs, ys, linewidth=1.2)
        plt.title(key)
        plt.xlabel("step")
        plt.ylabel(key)
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"{_sanitize(key)}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot W&B offline/online history metrics.")
    parser.add_argument("--history-file", default="", help="Path to wandb-history.jsonl")
    parser.add_argument("--root", default="data/outputs", help="Root dir to search for runs")
    parser.add_argument("--out-dir", default="plots", help="Output dir for plots")
    args = parser.parse_args()

    rows: List[dict] = []
    history_path = ""
    if args.history_file:
        history_path = args.history_file
        if not os.path.isfile(history_path):
            raise FileNotFoundError(f"history file not found: {history_path}")
        rows = _load_history(history_path)
    else:
        files = _find_history_files(args.root)
        if files:
            history_path = _latest_file(files)
            rows = _load_history(history_path)
        else:
            run_files = _find_wandb_run_files(args.root)
            if not run_files:
                raise FileNotFoundError("No wandb-history.jsonl or run-*.wandb files found.")
            run_path = _latest_file(run_files)
            rows = _load_history_from_wandb(run_path)
            history_path = run_path
    series = _collect_series(rows)
    if not series:
        raise RuntimeError(f"No numeric metrics found in {history_path}")

    _plot_all(series, args.out_dir)
    print(f"Plotted {len(series)} metrics from {history_path}")
    print(f"Saved plots to {args.out_dir}/")


if __name__ == "__main__":
    main()
