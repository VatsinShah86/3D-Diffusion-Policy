from typing import Dict, Tuple
import copy
import json
import math
from pathlib import Path

import numpy as np
import numcodecs
import torch

from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer


class EpisodeZarrDataset(BaseDataset):
    """Dataset loader for datasets stored as one `.zarr` directory per episode."""

    @staticmethod
    def _resolve_dataset_path(dataset_path: str) -> Path:
        path = Path(dataset_path).expanduser()
        if path.is_absolute() and path.exists():
            return path

        candidate_roots = [
            Path.cwd(),
            Path(__file__).resolve().parents[2],
            Path(__file__).resolve().parents[3],
        ]
        for root in candidate_roots:
            candidate = (root / path).resolve()
            if candidate.exists():
                return candidate

        return path

    @staticmethod
    def _normalize_state_keys(state_keys) -> Tuple[str, ...]:
        if isinstance(state_keys, str):
            state_keys = (state_keys,)
        else:
            state_keys = tuple(state_keys)

        if not state_keys:
            raise ValueError('state_keys must contain at least one observation key')
        return state_keys

    @staticmethod
    def _is_episode_path(path: Path) -> bool:
        return path.is_dir() and path.suffix == '.zarr' and (path / 'actions').exists()

    @classmethod
    def _candidate_episode_paths(cls, dataset_path: Path) -> Tuple[Path, ...]:
        if cls._is_episode_path(dataset_path):
            return (dataset_path,)

        if not dataset_path.is_dir():
            raise FileNotFoundError(f'Dataset path does not exist: {dataset_path}')

        return tuple(sorted(
            path for path in dataset_path.iterdir() if cls._is_episode_path(path)
        ))

    @classmethod
    def _collect_episode_paths(
            cls,
            dataset_path: Path,
            state_keys: Tuple[str, ...]) -> Tuple[Path, ...]:
        candidate_paths = cls._candidate_episode_paths(dataset_path)

        episode_paths = tuple(
            path for path in candidate_paths
            if cls._has_required_members(path=path, state_keys=state_keys)
        )
        if not episode_paths:
            raise FileNotFoundError(
                f'No loadable episode .zarr directories found under {dataset_path}'
            )
        return episode_paths

    @staticmethod
    def _has_required_members(path: Path, state_keys: Tuple[str, ...]) -> bool:
        required_paths = [
            path / 'actions' / 'action' / 'zarr.json',
            path / 'pointcloud' / 'zarr.json',
        ]
        required_paths.extend(
            path / 'observations' / state_key / 'zarr.json'
            for state_key in state_keys
        )
        return all(member.exists() for member in required_paths)

    @staticmethod
    def _load_v3_array(array_path: Path) -> np.ndarray:
        with open(array_path / 'zarr.json', 'r') as f:
            meta = json.load(f)

        if meta.get('zarr_format') != 3:
            raise ValueError(f'Expected Zarr v3 array at {array_path}')

        shape = tuple(meta['shape'])
        chunk_shape = tuple(meta['chunk_grid']['configuration']['chunk_shape'])
        dtype = np.dtype(meta['data_type'])
        endian = meta['codecs'][0]['configuration'].get('endian', 'little')
        dtype = dtype.newbyteorder('<' if endian == 'little' else '>')

        if any(chunk_dim != dim for chunk_dim, dim in zip(chunk_shape[1:], shape[1:])):
            raise ValueError(
                f'Unsupported chunking at {array_path}: '
                f'expected only the time dimension to be chunked, '
                f'got shape={shape}, chunk_shape={chunk_shape}'
            )

        data = np.empty(shape=shape, dtype=dtype)
        fill_value = meta.get('fill_value', 0.0)
        n_time_chunks = int(math.ceil(shape[0] / chunk_shape[0]))

        compressor = None
        for codec in meta['codecs']:
            if codec['name'] == 'zstd':
                zstd_config = dict(codec['configuration'])
                zstd_level = zstd_config.get('level', 0)
                compressor = numcodecs.Zstd(level=zstd_level)
                break
        if compressor is None:
            raise ValueError(f'Unsupported codec chain at {array_path}: {meta["codecs"]}')

        chunk_suffix = tuple('0' for _ in range(len(shape) - 1))
        for chunk_idx in range(n_time_chunks):
            chunk_path = array_path / 'c' / str(chunk_idx)
            for suffix in chunk_suffix:
                chunk_path = chunk_path / suffix

            time_start = chunk_idx * chunk_shape[0]
            time_end = min(time_start + chunk_shape[0], shape[0])

            if not chunk_path.exists():
                # Zarr v3 may omit chunks that are entirely equal to fill_value.
                data[time_start:time_end] = fill_value
                continue

            with open(chunk_path, 'rb') as f:
                encoded = f.read()

            decoded = compressor.decode(encoded)
            current_shape = (time_end - time_start,) + shape[1:]
            chunk = np.frombuffer(decoded, dtype=dtype).reshape(current_shape)
            data[time_start:time_end] = chunk

        return data.astype(np.float32, copy=False)

    @classmethod
    def _read_episode_arrays(
            cls,
            episode_path: Path,
            state_keys: Tuple[str, ...]) -> Dict[str, np.ndarray]:
        state_parts = []
        for state_key in state_keys:
            state_parts.append(
                cls._load_v3_array(episode_path / 'observations' / state_key)
            )

        if len(state_parts) == 1:
            state = state_parts[0]
        else:
            state = np.concatenate(state_parts, axis=-1)

        action = cls._load_v3_array(episode_path / 'actions' / 'action')
        point_cloud = cls._load_v3_array(episode_path / 'pointcloud')

        episode_length = state.shape[0]
        if action.shape[0] != episode_length or point_cloud.shape[0] != episode_length:
            raise ValueError(
                f'Length mismatch in {episode_path}: '
                f'state={state.shape[0]}, action={action.shape[0]}, '
                f'point_cloud={point_cloud.shape[0]}'
            )

        return {
            'state': state,
            'action': action,
            'point_cloud': point_cloud,
        }

    @classmethod
    def _build_replay_buffer(
            cls,
            dataset_path: Path,
            state_keys: Tuple[str, ...],
            skip_incomplete_episodes: bool) -> Tuple[ReplayBuffer, Tuple[Path, ...], Tuple[Path, ...]]:
        replay_buffer = ReplayBuffer.create_empty_numpy()
        loaded_episode_paths = []
        skipped_episode_paths = list(
            path for path in cls._candidate_episode_paths(dataset_path)
            if not cls._has_required_members(path=path, state_keys=state_keys)
        )

        for episode_path in cls._collect_episode_paths(dataset_path, state_keys):
            try:
                episode_data = cls._read_episode_arrays(
                    episode_path=episode_path,
                    state_keys=state_keys,
                )
            except Exception as exc:
                if not skip_incomplete_episodes:
                    raise RuntimeError(f'Failed to load episode {episode_path}') from exc
                skipped_episode_paths.append(episode_path)
                continue
            replay_buffer.add_episode(episode_data)
            loaded_episode_paths.append(episode_path)

        if replay_buffer.n_episodes == 0:
            raise RuntimeError(f'No valid episodes were loaded from {dataset_path}')

        if skipped_episode_paths and not skip_incomplete_episodes:
            skipped_paths_str = ', '.join(str(path) for path in tuple(skipped_episode_paths))
            raise RuntimeError(
                f'Found episode directories missing required arrays for state_keys={state_keys}: '
                f'{skipped_paths_str}'
            )

        return replay_buffer, tuple(loaded_episode_paths), tuple(skipped_episode_paths)

    def __init__(self,
            zarr_path,
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            task_name=None,
            state_keys=('ee_state',),
            skip_incomplete_episodes=True,
            augment=False,
            aug_xyz_sigma=0.003,
            aug_dropout_ratio=0.15,
            aug_dropout_frame_ratio=0.20,
            aug_color_sigma=8.0,
            ):
        super().__init__()
        self.task_name = task_name
        self.dataset_path = self._resolve_dataset_path(zarr_path)
        self.state_keys = self._normalize_state_keys(state_keys)
        self.skip_incomplete_episodes = skip_incomplete_episodes
        self.augment = augment
        self.aug_xyz_sigma = aug_xyz_sigma
        self.aug_dropout_ratio = aug_dropout_ratio
        self.aug_dropout_frame_ratio = aug_dropout_frame_ratio
        self.aug_color_sigma = aug_color_sigma

        (self.replay_buffer,
         self.episode_paths,
         self.skipped_episode_paths) = self._build_replay_buffer(
            dataset_path=self.dataset_path,
            state_keys=self.state_keys,
            skip_incomplete_episodes=skip_incomplete_episodes,
        )

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
        )
        val_set.train_mask = ~self.train_mask
        val_set.augment = False  # never augment validation data
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'][...,:],
            'point_cloud': self.replay_buffer['point_cloud'],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)

        # When the point cloud has a seg-id channel (index 6), it is a categorical
        # label in {0,1,2,3} and must NOT be limits-scaled. Force it to identity
        # (scale=1, offset=0) so the raw integer label passes through unchanged.
        # xyz (0:3) and rgb (3:6) keep their limits scaling.
        # For 6-col point clouds (no seg channel) this block is skipped entirely.
        SEG_IDX = 6
        pc_params = normalizer.params_dict['point_cloud']
        if len(pc_params['scale']) > SEG_IDX:
            with torch.no_grad():
                pc_params['scale'][SEG_IDX]  = 1.0
                pc_params['offset'][SEG_IDX] = 0.0

        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'].astype(np.float32))

    def __len__(self) -> int:
        return len(self.sampler)

    def _augment_point_cloud(self, pc: np.ndarray) -> np.ndarray:
        """Augment a (horizon, N, 7) point cloud sequence.

        Columns: xyz (0:3) | rgb (3:6) | seg_id (6) — seg_id is never touched.

        Global translation is sampled once and applied to every frame so that
        the implied camera motion across the horizon stays physically consistent.
        XYZ jitter, dropout, and color noise are sampled independently per frame.
        """
        pc = pc.copy()
        H, N, _ = pc.shape

        for t in range(H):
            # Per-point XYZ jitter
            if self.aug_xyz_sigma > 0.0:
                pc[t, :, :3] += np.random.normal(
                    0.0, self.aug_xyz_sigma, size=(N, 3)
                ).astype(np.float32)

            # Point dropout: applied only to a random subset of frames
            if self.aug_dropout_ratio > 0.0 and np.random.random() < self.aug_dropout_frame_ratio:
                n_drop = max(1, int(N * self.aug_dropout_ratio))
                drop_idx = np.random.choice(N, size=n_drop, replace=False)
                fill_idx = np.random.choice(N, size=n_drop, replace=True)
                pc[t, drop_idx, :6] = pc[t, fill_idx, :6]

            # Per-channel color noise, clipped to [0, 255]
            if self.aug_color_sigma > 0.0:
                noise = np.random.normal(
                    0.0, self.aug_color_sigma, size=(N, 3)
                ).astype(np.float32)
                pc[t, :, 3:6] = np.clip(pc[t, :, 3:6] + noise, 0.0, 255.0)

        return pc

    def _sample_to_data(self, sample):
        agent_pos = sample['state'].astype(np.float32)
        point_cloud = sample['point_cloud'].astype(np.float32)

        if self.augment:
            point_cloud = self._augment_point_cloud(point_cloud)

        data = {
            'obs': {
                'point_cloud': point_cloud,
                'agent_pos': agent_pos,
            },
            'action': sample['action'].astype(np.float32)
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
