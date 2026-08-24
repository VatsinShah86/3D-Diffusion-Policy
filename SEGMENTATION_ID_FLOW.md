# Segmentation ID Flow in DP3

This traces, line by line, how the per-point segmentation ID travels from the
raw point cloud through `dp3.py` into `DP3Encoder`, exactly how it reshapes
the point cloud's feature dimension, and what `id = 0` specifically does to
each point it's attached to.

All concrete shapes below are taken from the actual training config used for
the segmentation-aware pipeline — **not** made-up defaults:

- [`diffusion_policy_3d/config/dp3.yaml`](3D-Diffusion-Policy/diffusion_policy_3d/config/dp3.yaml) (policy config)
- [`diffusion_policy_3d/config/task/real_cloth_manip_ext_seg.yaml`](3D-Diffusion-Policy/diffusion_policy_3d/config/task/real_cloth_manip_ext_seg.yaml) (task config — `ext_seg` = external segmentation)

```yaml
# dp3.yaml (policy:)
use_pc_color: true
pointnet_type: "pointnet"
one_hot_seg: true
seg_channel_idx: 6
num_seg_classes: 4   # {0:padding, 1,2,3:real classes}
encoder_output_dim: 64
pointcloud_encoder_cfg:
  in_channels: 7        # overwritten at runtime, see §4
  out_channels: 64       # = encoder_output_dim
  use_layernorm: true
  final_norm: layernorm

# real_cloth_manip_ext_seg.yaml (task.shape_meta)
obs:
  point_cloud: { shape: [2048, 7] }   # N=2048 points, 7 columns
  agent_pos:   { shape: [7] }
action: { shape: [7] }
```

```yaml
# dataloader / dp3.yaml top level
dataloader.batch_size: 128
n_obs_steps: 5
horizon: 16
n_action_steps: 8
```

This task's `shape_meta.obs` has **no** `imagin_robot` key, so
`use_imagined_robot=False` for this pipeline — the imagined-robot point
concatenation step (§5) does not fire here; it's documented anyway since it's
part of the same code path.

Files involved:
- [`diffusion_policy_3d/policy/dp3.py`](3D-Diffusion-Policy/diffusion_policy_3d/policy/dp3.py)
- [`diffusion_policy_3d/model/vision/pointnet_extractor.py`](3D-Diffusion-Policy/diffusion_policy_3d/model/vision/pointnet_extractor.py)
- [`diffusion_policy_3d/dataset/episode_zarr_dataset.py`](3D-Diffusion-Policy/diffusion_policy_3d/dataset/episode_zarr_dataset.py)

---

## 1. Point cloud column layout

Raw point cloud tensor per point has 7 columns (`shape_meta.point_cloud.shape = [2048, 7]`):

```
[ x, y, z,  r, g, b,  seg_id ]
  0  1  2   3  4  5     6
```

`seg_id` is an integer label in `{0, 1, 2, 3}` (`num_seg_classes=4`: id `0` is
reserved padding, ids `1,2,3` are real semantic classes). This is set at
dataset-build time (the `ext_seg` zarr at
`/mnt/hdd8tb/vatsin/YOLO_SAM2_processed/real_dataset_cloth_ext_seg`), not in
`dp3.py`.

---

## 2. Normalizer leaves `seg_id` untouched

[`episode_zarr_dataset.py:297-318`](3D-Diffusion-Policy/diffusion_policy_3d/dataset/episode_zarr_dataset.py#L297-L318) (`get_normalizer`):

```python
SEG_IDX = 6
pc_params = normalizer.params_dict['point_cloud']
if len(pc_params['scale']) > SEG_IDX:
    with torch.no_grad():
        pc_params['scale'][SEG_IDX]  = 1.0
        pc_params['offset'][SEG_IDX] = 0.0
```

Every other channel (xyz, rgb) gets `limits` min/max scaling. Column 6 is
forced to `scale=1, offset=0` — an identity transform — so after
`self.normalizer.normalize(obs_dict)` runs in `dp3.py`
([`predict_action`](3D-Diffusion-Policy/diffusion_policy_3d/policy/dp3.py#L234) /
[`compute_loss`](3D-Diffusion-Policy/diffusion_policy_3d/policy/dp3.py#L336)), `points[..., 6]` is **still the raw
integer label**, not a normalized float.

Augmentation ([`_augment_point_cloud`](3D-Diffusion-Policy/diffusion_policy_3d/dataset/episode_zarr_dataset.py#L326-L359), active here since
`augment: true` in dp3.yaml) also only ever touches columns `:6` (xyz jitter,
dropout-copy, color noise) — column 6 (`seg_id`) is never modified.

---

## 3. Config flows from `DP3.__init__` into `DP3Encoder`

[`dp3.py:42-44`](3D-Diffusion-Policy/diffusion_policy_3d/policy/dp3.py#L42-L44) — constructor args (values shown are what's actually
passed by `dp3.yaml`):

```python
one_hot_seg=True,
seg_channel_idx=6,
num_seg_classes=4,
```

[`dp3.py:83-92`](3D-Diffusion-Policy/diffusion_policy_3d/policy/dp3.py#L83-L92) — passed straight through to the encoder, no
transformation happens in `DP3` itself:

```python
obs_encoder = DP3Encoder(observation_space=obs_dict,
                          img_crop_shape=crop_shape,
                          out_channel=encoder_output_dim,   # 64
                          pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                          use_pc_color=use_pc_color,        # True
                          pointnet_type=pointnet_type,       # "pointnet"
                          one_hot_seg=one_hot_seg,           # True
                          seg_channel_idx=seg_channel_idx,   # 6
                          num_seg_classes=num_seg_classes,   # 4
                          )
```

So `dp3.py` is purely a pass-through for segmentation config. **All actual
segmentation-ID processing happens inside `DP3Encoder.forward`.**

---

## 4. `DP3Encoder.__init__` — sizing the encoder for one-hot expansion

[`pointnet_extractor.py:262-288`](3D-Diffusion-Policy/diffusion_policy_3d/model/vision/pointnet_extractor.py#L262-L288):

```python
if one_hot_seg and not use_pc_color:
    raise ValueError("one_hot_seg=True requires use_pc_color=True")
...
if use_pc_color:
    pointcloud_feature_dim = self.point_cloud_shape[-1]   # 7 (from shape_meta)
    ...
    if one_hot_seg:
        # scalar seg channel → num_seg_classes one-hot channels
        effective_channels = pointcloud_feature_dim - 1 + num_seg_classes
    else:
        effective_channels = pointcloud_feature_dim
    pointcloud_encoder_cfg.in_channels = effective_channels
    self.extractor = PointNetEncoderXYZRGB(**pointcloud_encoder_cfg)
```

With this task's real values (`pointcloud_feature_dim=7`, `num_seg_classes=4`):

```
effective_channels = 7 - 1 + 4 = 10
```

The `in_channels: 7` set in `dp3.yaml`'s `pointcloud_encoder_cfg` is
overwritten here — the per-point MLP (`PointNetEncoderXYZRGB`) is actually
built to accept **10** input channels: `[x,y,z, r,g,b, onehot_0, onehot_1, onehot_2, onehot_3]`.

`out_channels=64` (`encoder_output_dim`), so `self.extractor` maps
`(B, N, 10) → (B, 64)` per forward call (point-feature MLP + max-pool +
final projection to 64-d, with LayerNorm since `final_norm: layernorm`).

`agent_pos` is `shape: [7]`, run through `state_mlp` (default `state_mlp_size=(64,64)` —
no override in `dp3.yaml`) to a 64-d vector, so:

```
n_output_channels = encoder_output_dim + state_mlp_output_dim = 64 + 64 = 128
```

— this is `DP3Encoder.output_shape()`, which feeds `obs_feature_dim` in `dp3.py`.

---

## 5. `DP3Encoder.forward` — the exact per-point transformation

[`pointnet_extractor.py:307-334`](3D-Diffusion-Policy/diffusion_policy_3d/model/vision/pointnet_extractor.py#L307-L334):

```python
def forward(self, observations: Dict) -> torch.Tensor:
    points = observations[self.point_cloud_key]
    assert len(points.shape) == 3   # (B, N, 7) = (B, 2048, 7)

    valid_mask = None
    if self.one_hot_seg:
        seg = points[..., self.seg_channel_idx]                 # (B, N)  raw label, column 6
        seg_idx = seg.round().clamp(0, self.num_seg_classes - 1).long()
        one_hot = F.one_hot(seg_idx, num_classes=self.num_seg_classes).float()  # (B, N, 4)
        points = torch.cat([points[..., :self.seg_channel_idx], one_hot], dim=-1)  # (B, N, 6) + (B, N, 4) -> (B, N, 10)
        valid_mask = seg_idx != 0    # (B, N) bool — id 0 -> False, ids {1,2,3} -> True

    if self.use_imagined_robot:   # False for real_cloth_manip_ext_seg (no imagin_robot key)
        img_points = observations[self.imagination_key][..., :points.shape[-1]]
        points = torch.concat([points, img_points], dim=1)   # concat along N (more points), not feature dim
        if valid_mask is not None:
            B = valid_mask.shape[0]
            n_img = img_points.shape[1]
            img_valid = torch.ones((B, n_img), dtype=valid_mask.dtype, device=valid_mask.device)
            valid_mask = torch.cat([valid_mask, img_valid], dim=1)  # imagined points always treated as real

    pn_feat = self.extractor(points, valid_mask=valid_mask)   # (B, 64)
    ...
```

The effective batch dimension `B` here is `batch_size * n_obs_steps = 128 * 5 = 640`
(the `dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))` call
in `dp3.py`'s `predict_action`/`compute_loss` flattens the obs-history dim
into the batch before calling the encoder).

Step by step, on a `(640, 2048, 7)` input:

| Step | Operation | Shape before | Shape after |
|---|---|---|---|
| Extract seg column | `points[..., 6]` | `(640, 2048, 7)` | `(640, 2048)` |
| Round + clamp to valid range | `.round().clamp(0, 3)` | `(640, 2048)` float | `(640, 2048)` int in `{0,1,2,3}` |
| One-hot encode | `F.one_hot(seg_idx, 4).float()` | `(640, 2048)` | `(640, 2048, 4)` |
| Drop scalar seg column, append one-hot | `cat([points[...,:6], one_hot], dim=-1)` | `(640, 2048, 7)` | `(640, 2048, 10)` |
| Build validity mask | `seg_idx != 0` | — | `(640, 2048)` bool |
| (n/a here) append imagined-robot points | `cat([points, img_points], dim=1)` | `(640, 2048, 10)` | would be `(640, 2048+n_img, 10)` if enabled — grows the **point** dim, not the feature dim |

So **the feature/channel dimension of the point cloud grows from 7 → 10**
(scalar seg replaced by a 4-wide one-hot), while the point-count dimension
stays at `2048` for this task (no imagined-robot points configured).

---

## 6. What `id = 0` specifically does

`seg_id == 0` is the reserved **padding label**. Concretely, for a point whose
raw `seg_id` is `0`:

1. **One-hot encoding**: `seg_idx.round().clamp(0, 3)` keeps it at `0`, so
   `F.one_hot` produces `[1, 0, 0, 0]` — it activates one-hot channel index 0
   like any other class would for its index. The one-hot itself doesn't
   special-case `0`; it just encodes it like any class.
2. **Validity mask**: `valid_mask = seg_idx != 0` evaluates to `False` for
   this point. This is the actual special-casing — `id=0` is the *only*
   value among `{0,1,2,3}` that gets `False`.
3. **Masked out of the symmetric max-pool**: in
   [`PointNetEncoderXYZRGB.forward`](3D-Diffusion-Policy/diffusion_policy_3d/model/vision/pointnet_extractor.py#L108-L123):

   ```python
   x = self.mlp(x)                                   # (640, 2048, 512) per-point features
   if valid_mask is not None:
       neg_inf = torch.finfo(x.dtype).min
       x = x.masked_fill(~valid_mask.unsqueeze(-1), neg_inf)   # id=0 points -> -inf
   x = torch.max(x, 1)[0]                             # symmetric max-pool over N=2048
   x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
   x = self.final_projection(x)                       # -> (640, 64), LayerNorm applied
   ```

   A point with `seg_id=0` still runs through the per-point MLP (it's not
   skipped/removed from the tensor — `N` stays `2048`), but right before the
   max-pool every one of its 512 feature channels is overwritten with `-inf`
   (the minimum representable value of that dtype). Since the pooling
   operation is `max` over the point dimension, a feature equal to `-inf`
   can never win the max unless **every** point at that channel is `-inf`
   (i.e. the whole cloud is padding) — in which case the post-pool value is
   `-inf`/NaN and gets zeroed by `nan_to_num`.

   Net effect: **`seg_id=0` points contribute nothing to the pooled global
   point-cloud feature.** They exist in the tensor (so batch shapes stay
   fixed at `N=2048`), but are functionally invisible to the encoder
   output — this is how a fixed-size `2048`-point cloud handles a variable
   number of "real" points: pad with `seg_id=0` dummy points, and mask them
   out of the pooling step.

4. (Not exercised for this task, since `use_imagined_robot=False`): imagined-robot
   points appended afterward are always forced `valid_mask=True` regardless
   of any seg id they might carry, since they are synthetic and always
   "real" for the purposes of the encoder.

---

## 7. Summary diagram

```
points (B=640, N=2048, 7)        xyz | rgb | seg_id ∈ {0,1,2,3}
        │
        │ one_hot_seg = True
        ▼
seg = points[...,6]                      (640, 2048)
seg_idx = round().clamp(0, 3).long()      (640, 2048)        id=0 stays 0 (clamp floor)
one_hot = F.one_hot(seg_idx, 4)           (640, 2048, 4)      e.g. [1,0,0,0] for id 0
points  = cat(points[...,:6], one_hot)    (640, 2048, 10)     7 -> 10 channels
valid_mask = (seg_idx != 0)               (640, 2048)         id=0 -> False, else True
        │
        ▼ (use_imagined_robot=False for this task -> no-op here)
points (640, 2048, 10)
        │
        ▼ PointNetEncoderXYZRGB.forward  (in_channels=10, out_channels=64)
per-point MLP -> (640, 2048, 512)
masked_fill(~valid_mask, -inf)   # id=0 points zeroed out of contention
max-pool over N=2048             # id=0 points can't win unless ALL points in that cloud are id=0
nan_to_num                       # all-padding cloud -> 0 vector
final_projection (+LayerNorm) -> (640, 64)
        │
        ▼ concat with state_mlp(agent_pos) (640, 64)
final_feat (640, 128)            # = obs_feature_dim, fed into the diffusion U-Net as global_cond
```
