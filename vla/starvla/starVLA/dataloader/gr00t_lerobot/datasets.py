# Upstream attribution: the named implementation credits and retained
# legacy remarks in this file come from public StarVLA source/history
# (https://github.com/starVLA/starVLA), including revision
# f18fbc22c317dd1810839cb621632ac45add93f1 where applicable. They identify
# upstream contributions, not the authors or affiliations of this submission.

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
In this file, we define 3 types of datasets:
1. LeRobotSingleDataset: a single dataset for a given embodiment tag
2. LeRobotMixtureDataset: a mixture of datasets for a given list of embodiment tags
3. CachedLeRobotSingleDataset: a single dataset for a given embodiment tag,
                                with caching for the video frames

See `scripts/load_dataset.py` for examples on how to use these datasets.
"""

import gc
import hashlib
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

import os
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image

from starVLA.dataloader.gr00t_lerobot.video import get_all_frames, get_frames_by_timestamps

from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.schema import (
    DatasetMetadata,
    DatasetStatisticalValues,
    LeRobotModalityMetadata,
    LeRobotStateActionMetadata,
)
from starVLA.dataloader.gr00t_lerobot.transform import (
    ComposedModalityTransform, 
    RelativeActionTransform,
    calculate_relative_dataset_statistics,
)

from functools import partial
from typing import Tuple, List
import pickle

LE_ROBOT_MODALITY_FILENAME = "meta/modality.json"
LE_ROBOT_EPISODE_FILENAME = "meta/episodes.jsonl"
LE_ROBOT_TASKS_FILENAME = "meta/tasks.jsonl"
LE_ROBOT_TASKS_REWRITTEN_FILENAME = "meta/tasks_rewritten.jsonl"
LE_ROBOT_TASKS_SIMPLE_FILENAME = "meta/tasks_simple_v2.jsonl"
LE_ROBOT_INFO_FILENAME = "meta/info.json"
##original
# LE_ROBOT_STATS_FILENAME = "meta/stats_gr00t.json"
LE_ROBOT_STATS_FILENAME = "meta/stats.json"
LE_ROBOT_STATS_DELTA_FILENAME = "meta/stats_delta.json"
LE_ROBOT_DATA_FILENAME = "data/*/*.parquet"

# Default action chunk size for relative stats computation
DEFAULT_ACTION_CHUNK_SIZE = 50


def get_relative_stats_filename(relative_type: str, frame_type: str, action_chunk_size: int = DEFAULT_ACTION_CHUNK_SIZE) -> str:
    """Generate the stats filename for relative action transforms.
    
    Args:
        relative_type: "relative" or "delta"
        frame_type: "local" or "global"
        action_chunk_size: Action chunk size. If not default (50), appends "_chunk{size}" to filename.
        
    Returns:
        Filename like "meta/stats_relative_local.json" or "meta/stats_delta_local_chunk16.json"
    """
    base_name = f"stats_{relative_type}_{frame_type}"
    if action_chunk_size != DEFAULT_ACTION_CHUNK_SIZE:
        base_name = f"{base_name}_chunk{action_chunk_size}"
    return f"meta/{base_name}.json"

LE_ROBOT_STEPS_FILENAME = "meta/steps.pkl"
EPSILON = 5e-4

def calculate_dataset_statistics(parquet_paths: list[Path]) -> dict:
    """Calculate the dataset statistics of all columns for a list of parquet files."""
    from multiprocessing import cpu_count
    n_threads = max(1, cpu_count())
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        all_low_dim_data_list = list(tqdm(
            executor.map(pd.read_parquet, sorted(parquet_paths)),
            total=len(parquet_paths),
            desc=f"Collecting all parquet files ({n_threads} threads)...",
        ))
    all_low_dim_data = pd.concat(all_low_dim_data_list, axis=0)
    # Compute dataset statistics
    dataset_statistics = {}
    for le_modality in all_low_dim_data.columns:
        if le_modality.startswith("annotation."):
            continue
        print(f"Computing statistics for {le_modality}...")
        np_data = np.vstack(
            [np.asarray(x, dtype=np.float32) for x in all_low_dim_data[le_modality]]
        )
        dataset_statistics[le_modality] = {
            "mean": np.mean(np_data, axis=0).tolist(),
            "std": np.std(np_data, axis=0).tolist(),
            "min": np.min(np_data, axis=0).tolist(),
            "max": np.max(np_data, axis=0).tolist(),
            "q01": np.quantile(np_data, 0.01, axis=0).tolist(),
            "q99": np.quantile(np_data, 0.99, axis=0).tolist(),
        }
    return dataset_statistics


class ModalityConfig(BaseModel):
    """Configuration for a modality."""

    delta_indices: list[int]
    """Delta indices to sample relative to the current index. The returned data will correspond to the original data at a sampled base index + delta indices."""
    modality_keys: list[str]
    """The keys to load for the modality in the dataset."""


class LeRobotSingleDataset(Dataset):
    """
    Base dataset class for LeRobot that supports sharding.
    """
    def __init__(
        self,
        dataset_path: Path | str,
        modality_configs: dict[str, ModalityConfig],
        embodiment_tag: str | EmbodimentTag,
        video_backend: str = "decord",
        video_backend_kwargs: dict | None = None,
        transforms: ComposedModalityTransform | None = None,
        delete_pause_frame: bool = False,
        num_shot: int = None,
        data_percent: float | None = None,
        subset_seed: int = 42,
        use_delta: bool = False,
        skip_stats_computation: bool = False,
        use_simple_tasks: bool = False,
        target_fps: int | None = None,
        info_filename: str | None = None,
        modality_filename: str | None = None,
        use_episode_instruction: bool = True,
        strip_bilingual_task_prefix: bool = False,
        stats_cache_dir: Path | str | None = None,
    ):
        """
        Initialize the dataset.

        Args:
            dataset_path (Path | str): The path to the dataset.
            modality_configs (dict[str, ModalityConfig]): The configuration for each modality. The keys are the modality names, and the values are the modality configurations.
                See `ModalityConfig` for more details.
            video_backend (str): Backend for video reading.
            video_backend_kwargs (dict): Keyword arguments for the video backend when initializing the video reader.
            transforms (ComposedModalityTransform): The transforms to apply to the dataset.
            embodiment_tag (EmbodimentTag): Overload the embodiment tag for the dataset. e.g. define it as "new_embodiment"
            use_simple_tasks (bool): If True, use tasks_simple_v2.jsonl for task augmentation during training.
                Each task in tasks_simple_v2.jsonl should have a list of task variations, where the first element
                matches the original task from get_language. A random task will be chosen from the list.
            info_filename (str | None): Custom filename under meta/ for info (e.g. "info_processed.json").
                If None, uses the default "meta/info.json".
            modality_filename (str | None): Custom filename under meta/ for modality (e.g. "modality_processed.json").
                If None, uses the default "meta/modality.json".
            use_episode_instruction (bool): If False, ignore per-episode task annotations from episodes.jsonl
                and fall back to per-step task_index via tasks.jsonl.
            strip_bilingual_task_prefix (bool): If True, strip Chinese text before '@' in task strings
                (e.g. "中文描述@English description" -> "English description").
            stats_cache_dir (Path | str | None): Writable directory for caching computed stats files.
                When set, stats are read from / written to a mirror of the dataset path under this
                directory, avoiding writes to potentially read-only dataset directories.
        """
        # first check if the path directory exists
        if not Path(dataset_path).exists():
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")

        self.delete_pause_frame = delete_pause_frame

        self.modality_configs = modality_configs
        self.video_backend = video_backend
        print("self.video_backend", self.video_backend)
        self.video_backend_kwargs = video_backend_kwargs if video_backend_kwargs is not None else {}
        self.transforms = (
            transforms if transforms is not None else ComposedModalityTransform(transforms=[])
        )

        self._dataset_path = Path(dataset_path)
        self._dataset_name = self._dataset_path.name
        self.skip_stats_computation = skip_stats_computation
        self._stats_cache_dir = Path(stats_cache_dir) if stats_cache_dir is not None else None
        self._info_filename = info_filename
        self._modality_filename = modality_filename
        self._use_episode_instruction = use_episode_instruction
        self.strip_bilingual_task_prefix = strip_bilingual_task_prefix
        if isinstance(embodiment_tag, EmbodimentTag):
            self.tag = embodiment_tag.value
        else:
            self.tag = embodiment_tag

        print(embodiment_tag, isinstance(embodiment_tag, EmbodimentTag))
        print("self.tag ", self.tag, EmbodimentTag(self.tag))

        self._metadata = self._get_metadata(EmbodimentTag(self.tag), use_delta)

        # LeRobot-specific config
        self._lerobot_modality_meta = self._get_lerobot_modality_meta()
        self._lerobot_info_meta = self._get_lerobot_info_meta()
        self._data_path_pattern = self._get_data_path_pattern()
        self._video_path_pattern = self._get_video_path_pattern()
        self._chunk_size = self._get_chunk_size()
        self._tasks = self._get_tasks()
        self.curr_traj_data = None
        self.curr_traj_id = None
        self.num_shot = num_shot
        self.data_percent = data_percent
        self.subset_seed = int(subset_seed)
        if self.data_percent is not None and self.num_shot is not None:
            raise ValueError("data_percent and num_shot are mutually exclusive")
        if self.data_percent is not None and not (0 < float(self.data_percent) <= 100):
            raise ValueError(f"data_percent must be in (0, 100], got {self.data_percent}")
        self.use_simple_tasks = use_simple_tasks

        self.source_fps: int = self._lerobot_info_meta["fps"]
        self.target_fps: int | None = target_fps
        if self.target_fps is not None and self.target_fps != self.source_fps:
            print(f"FPS resampling enabled: {self.source_fps} -> {self.target_fps} for {self._dataset_name}")
        
        # Load simple tasks mapping if enabled
        self._simple_tasks_mapping: dict[str, list[str]] | None = None
        if self.use_simple_tasks:
            self._simple_tasks_mapping = self._load_simple_tasks_mapping()

        ## original code
        #self._trajectory_ids, self._trajectory_lengths = self._get_trajectories()
        ## for support gr1 language instruction (key as "remark")
        self._trajectory_ids, self._trajectory_lengths, self._trajectory_tasks = self._get_trajectories()
        if not self._use_episode_instruction:
            self._trajectory_tasks = None
        self._modality_keys = self._get_modality_keys()
        self._delta_indices = self._get_delta_indices()
        self._all_steps = self._get_all_steps()

        self.set_transforms_metadata(self.metadata)
        self.set_epoch(0)

        print(f"Initialized dataset {self.dataset_name} with {embodiment_tag}")

        # Check if the dataset is valid
        self._check_integrity()

    @property
    def dataset_path(self) -> Path:
        """The path to the dataset that contains the METADATA_FILENAME file."""
        return self._dataset_path

    @property
    def metadata(self) -> DatasetMetadata:
        """The metadata for the dataset, loaded from metadata.json in the dataset directory"""
        return self._metadata

    @property
    def trajectory_tasks(self) -> list[str]:
        return self._trajectory_tasks
        
    @property
    def trajectory_ids(self) -> np.ndarray:
        """The trajectory IDs in the dataset, stored as a 1D numpy array of strings."""
        return self._trajectory_ids

    @property
    def trajectory_lengths(self) -> np.ndarray:
        """The trajectory lengths in the dataset, stored as a 1D numpy array of integers.
        The order of the lengths is the same as the order of the trajectory IDs.
        """
        return self._trajectory_lengths

    @property
    def all_steps(self) -> list[tuple[int, int]]:
        """The trajectory IDs and base indices for all steps in the dataset.
        Example:
            self.trajectory_ids: [0, 1, 2]
            self.trajectory_lengths: [3, 2, 4]
            return: [
                ("traj_0", 0), ("traj_0", 1), ("traj_0", 2),
                ("traj_1", 0), ("traj_1", 1),
                ("traj_2", 0), ("traj_2", 1), ("traj_2", 2), ("traj_2", 3)
            ]
        """
        return self._all_steps

    @property
    def modality_keys(self) -> dict:
        """The modality keys for the dataset. The keys are the modality names, and the values are the keys for each modality.

        Example: {
            "video": ["video.image_side_0", "video.image_side_1"],
            "state": ["state.eef_position", "state.eef_rotation"],
            "action": ["action.eef_position", "action.eef_rotation"],
            "language": ["language.human.task"],
            "timestamp": ["timestamp"],
            "reward": ["reward"],
        }
        """
        return self._modality_keys

    @property
    def delta_indices(self) -> dict[str, np.ndarray]:
        """The delta indices for the dataset. The keys are the modality.key, and the values are the delta indices for each modality.key."""
        return self._delta_indices

    @property
    def dataset_name(self) -> str:
        """The name of the dataset."""
        return self._dataset_name

    @property
    def lerobot_modality_meta(self) -> LeRobotModalityMetadata:
        """The metadata for the LeRobot dataset."""
        return self._lerobot_modality_meta

    @property
    def lerobot_info_meta(self) -> dict:
        """The metadata for the LeRobot dataset."""
        return self._lerobot_info_meta

    @property
    def data_path_pattern(self) -> str:
        """The path pattern for the LeRobot dataset."""
        return self._data_path_pattern

    @property
    def video_path_pattern(self) -> str:
        """The path pattern for the LeRobot dataset."""
        return self._video_path_pattern

    @property
    def chunk_size(self) -> int:
        """The chunk size for the LeRobot dataset."""
        return self._chunk_size

    @property
    def tasks(self) -> pd.DataFrame:
        """The tasks for the dataset."""
        return self._tasks

    @property
    def simple_tasks_mapping(self) -> dict[str, list[str]] | None:
        """The simple tasks mapping for task augmentation. 
        Maps main task string to list of task variations."""
        return self._simple_tasks_mapping

    def _get_relative_transform(self) -> RelativeActionTransform | None:
        """Get the RelativeActionTransform if present in transforms."""
        if self.transforms is None:
            return None
        for transform in self.transforms.transforms:
            if isinstance(transform, RelativeActionTransform):
                return transform
        return None

    def _get_stats_path(self, use_delta: bool) -> Path:
        """Get the appropriate stats file path based on transform configuration."""
        transform = self._get_relative_transform()
        if transform:
            filename = get_relative_stats_filename(
                transform.relative_type, 
                transform.frame_type, 
                transform.action_chunk_size
            )
            return self.dataset_path / filename
        return self.dataset_path / (LE_ROBOT_STATS_DELTA_FILENAME if use_delta else LE_ROBOT_STATS_FILENAME)

    def _get_cache_path(self, stats_path: Path) -> Path | None:
        """Map a dataset-relative stats path to the cache directory."""
        if self._stats_cache_dir is None:
            return None
        # Mirror the absolute dataset stats path under cache dir (strip leading '/')
        return self._stats_cache_dir / str(stats_path).lstrip("/")

    def _load_or_compute_statistics(self, stats_path: Path, modality_meta=None, data_path_pattern: str | None = None) -> dict:
        """Load statistics from file, cache dir, or compute if not found.

        When stats_cache_dir is set, we ONLY use the cache dir:
          - If cached file exists → load it.
          - If not → compute, write to cache dir, return.
          - The original stats_path in the dataset is completely ignored.

        When stats_cache_dir is NOT set, we use the original stats_path
        (load if exists, compute and write there otherwise).
        """
        cache_path = self._get_cache_path(stats_path)

        if cache_path is not None:
            # Cache dir mode: only use cache_path, ignore original stats_path
            if cache_path.exists():
                with open(cache_path, "r") as f:
                    stats = json.load(f)
                print(f"Loaded cached statistics from {cache_path}")
                return stats
            # Not in cache → compute
        else:
            # No cache dir: use original stats_path
            if stats_path.exists():
                with open(stats_path, "r") as f:
                    stats = json.load(f)
                return stats

        # Compute statistics
        print(f"Computing statistics for {stats_path}...")
        import re
        if data_path_pattern is not None:
            glob_pattern = re.sub(r'\{[^}]+\}', '*', data_path_pattern)
        else:
            glob_pattern = LE_ROBOT_DATA_FILENAME
        parquet_files = list(self.dataset_path.glob(glob_pattern))

        transform = self._get_relative_transform()
        if transform:
            stats = calculate_relative_dataset_statistics(
                parquet_files,
                modality_meta,
                transform.relative_type,
                transform.frame_type,
                action_chunk_size=transform.action_chunk_size,
            )
        else:
            stats = calculate_dataset_statistics(parquet_files)

        write_path = cache_path if cache_path is not None else stats_path
        write_path.parent.mkdir(parents=True, exist_ok=True)
        with open(write_path, "w") as f:
            json.dump(stats, f, indent=4)
        print(f"Saved statistics to {write_path}")
        return stats

    def _get_metadata(self, embodiment_tag: EmbodimentTag, use_delta: bool) -> DatasetMetadata:
        """Get the metadata for the dataset.

        Returns:
            dict: The metadata for the dataset.
        """

        # 1. Modality metadata
        if self._modality_filename is not None:
            modality_meta_path = self.dataset_path / "meta" / self._modality_filename
        else:
            modality_meta_path = self.dataset_path / LE_ROBOT_MODALITY_FILENAME
        assert (
            modality_meta_path.exists()
        ), f"Please provide {modality_meta_path} (modality metadata) in {self.dataset_path}"
        # 1.1. State and action modalities
        simplified_modality_meta: dict[str, dict] = {}
        with open(modality_meta_path, "r") as f:
            le_modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))
        for modality in ["state", "action"]:
            simplified_modality_meta[modality] = {}
            le_state_action_meta: dict[str, LeRobotStateActionMetadata] = getattr(
                le_modality_meta, modality
            )
            for subkey in le_state_action_meta:
                state_action_dtype = np.dtype(le_state_action_meta[subkey].dtype)
                if np.issubdtype(state_action_dtype, np.floating):
                    continuous = True
                else:
                    continuous = False
                simplified_modality_meta[modality][subkey] = {
                    "absolute": le_state_action_meta[subkey].absolute,
                    "rotation_type": le_state_action_meta[subkey].rotation_type,
                    "shape": [
                        le_state_action_meta[subkey].end - le_state_action_meta[subkey].start
                    ],
                    "continuous": continuous,
                }
        # 1.2. Video modalities
        if self._info_filename is not None:
            le_info_path = self.dataset_path / "meta" / self._info_filename
        else:
            le_info_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        assert le_info_path.exists(), f"Please provide {le_info_path} (info metadata) in {self.dataset_path}"
        with open(le_info_path, "r") as f:
            le_info = json.load(f)
        simplified_modality_meta["video"] = {}
        for new_key in le_modality_meta.video:
            original_key = le_modality_meta.video[new_key].original_key
            if original_key is None:
                original_key = new_key
            le_video_meta = le_info["features"][original_key]
            height = le_video_meta["shape"][le_video_meta["names"].index("height")]
            width = le_video_meta["shape"][le_video_meta["names"].index("width")]
            # NOTE(FH): different lerobot dataset versions have different keys for the number of channels and fps
            try:
                channels = le_video_meta["shape"][le_video_meta["names"].index("channel")]
                fps = le_video_meta["video_info"]["video.fps"]
            except (ValueError, KeyError):
                # channels = le_video_meta["shape"][le_video_meta["names"].index("channels")]
                channels = le_video_meta["info"]["video.channels"]
                fps = le_video_meta["info"]["video.fps"]
            # If decord video_backend_kwargs specifies height/width for decode-time resize,
            # use the decode size instead of the original video size so downstream shape
            # checks (e.g. VideoToTensor, VideoCrop) match the actual decoded frame size.
            decord_width = self.video_backend_kwargs.get("width", None)
            decord_height = self.video_backend_kwargs.get("height", None)
            if decord_width is not None:
                width = decord_width
            if decord_height is not None:
                height = decord_height

            simplified_modality_meta["video"][new_key] = {
                "resolution": [width, height],
                "channels": channels,
                "fps": fps,
            }
        # 2. Dataset statistics - just pick the right file based on transform config
        stats_path = self._get_stats_path(use_delta)
        le_statistics = self._load_or_compute_statistics(
            stats_path, le_modality_meta,
            data_path_pattern=le_info.get("data_path"),
        )
        
        # Build dataset_statistics by extracting indices from the loaded stats.
        # Relative stats use modality-prefixed keys (_action_stats/col, _state_stats/col)
        # to avoid collisions when action and state sub-keys share the same original_key.
        # Non-relative stats use plain column names.
        is_relative_stats = "_relative_info" in le_statistics
        dataset_statistics = {}
        for our_modality in ["state", "action"]:
            dataset_statistics[our_modality] = {}
            for subkey in simplified_modality_meta[our_modality]:
                full_key = f"{our_modality}.{subkey}"
                state_action_meta = le_modality_meta.get_key_meta(full_key)
                assert isinstance(state_action_meta, LeRobotStateActionMetadata)
                
                le_key = state_action_meta.original_key
                indices = np.arange(state_action_meta.start, state_action_meta.end)
                
                if is_relative_stats:
                    stats_key = f"_{our_modality}_stats/{le_key}"
                else:
                    stats_key = le_key
                assert stats_key in le_statistics, (
                    f"Stats key '{stats_key}' not found. "
                    f"Available: {sorted(k for k in le_statistics if not k.startswith('_'))}"
                )
                column_stats = le_statistics[stats_key]
                
                subkey_stats = {}
                for stat_name in column_stats:
                    stat_array = np.array(column_stats[stat_name])
                    if stat_array.ndim == 1:
                        subkey_stats[stat_name] = stat_array[indices].tolist()
                    else:
                        subkey_stats[stat_name] = stat_array[:, indices].tolist()
                dataset_statistics[our_modality][subkey] = subkey_stats
        #print("dataset_statistics", dataset_statistics)
        # 3. Full dataset metadata
        metadata = DatasetMetadata(
            statistics=dataset_statistics,  # type: ignore
            modalities=simplified_modality_meta,  # type: ignore
            embodiment_tag=embodiment_tag,
        )

        return metadata

    def _get_trajectories(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the trajectories in the dataset."""
        # Get trajectory lengths, IDs, and whitelist from dataset metadata
        episode_path = self.dataset_path / LE_ROBOT_EPISODE_FILENAME
        with open(episode_path, "r", encoding="utf-8") as f:
            episode_metadata = [json.loads(line) for line in f]
        trajectory_ids = []
        trajectory_lengths = []
        trajectory_tasks = []
        has_episode_tasks = True
        for episode in episode_metadata:
            trajectory_ids.append(episode["episode_index"])
            trajectory_lengths.append(episode["length"])

            if "remarks" in episode.keys(): # Only for PhysicalAI-Robotics-GR00T-Teleop-Sim Data
                trajectory_tasks.append(["unlocked_waist: " + episode["remarks"]])
            elif "tasks" in episode:
                # datasets whose episodes carry a `tasks` field use it as the instruction
                trajectory_tasks.append(episode["tasks"])
            else:
                has_episode_tasks = False
        if not has_episode_tasks:
            trajectory_tasks = None

        # # NOTE: Only for PhysicalAI-Robotics-GR00T-Teleop-Sim Data
        # if len(episode_metadata[0]["tasks"][0].split()) == 1:
        #     trajectory_tasks = []
        #     for episode in episode_metadata:
        #         trajectory_tasks.append(["unlocked_waist: " + episode["remarks"]])
        #     print(f"Using unified task instructions, e.g., {episode_metadata[0]['tasks']} --> {trajectory_tasks[0]}")
        # else:
        #     trajectory_tasks = None

        if self.data_percent is not None:
            # Select the requested percentage independently inside every dataset/task.
            # A stable per-dataset seed makes the subsets reproducible and nested:
            # 25% is a prefix of 50%, which is a prefix of 75% and 100%.
            total = len(trajectory_ids)
            selected_count = max(1, int(np.floor(total * float(self.data_percent) / 100.0)))
            dataset_seed = int.from_bytes(
                hashlib.sha256(self._dataset_name.encode("utf-8")).digest()[:8], "little"
            ) ^ self.subset_seed
            order = np.random.default_rng(dataset_seed).permutation(total)[:selected_count]
            trajectory_ids = [trajectory_ids[i] for i in order]
            trajectory_lengths = [trajectory_lengths[i] for i in order]
            trajectory_tasks = (
                [trajectory_tasks[i] for i in order]
                if trajectory_tasks is not None
                else None
            )
            print(
                f"[data subset] task={self._dataset_name} percent={self.data_percent:g} "
                f"episodes={selected_count}/{total} seed={self.subset_seed}"
            )
        elif self.num_shot is not None:
            trajectory_ids = trajectory_ids[:self.num_shot]
            trajectory_lengths = trajectory_lengths[:self.num_shot]
            trajectory_tasks = trajectory_tasks[:self.num_shot] if trajectory_tasks is not None else trajectory_tasks

        return np.array(trajectory_ids), np.array(trajectory_lengths), trajectory_tasks


    ## do not use data cache here 
    def _get_all_steps(self) -> list[tuple[int, int]]:
        """Get the trajectory IDs and base indices for all steps in the dataset.

        Returns:
            list[tuple[str, int]]: A list of (trajectory_id, base_index) tuples.
        """
        # Compute steps using single process
        all_steps = self._get_all_steps_single_process()
        
        return all_steps

    ## original code for data cache
    # def _get_all_steps(self) -> list[tuple[int, int]]:
    #     """Get the trajectory IDs and base indices for all steps in the dataset.

    #     Returns:
    #         list[tuple[str, int]]: A list of (trajectory_id, base_index) tuples.
    #     """
    #     # Create a hash key based on configuration to ensure cache validity
    #     config_key = self._get_steps_config_key()
        
    #     # Create a unique filename based on config_key
    #     steps_filename = f"steps_{config_key}.pkl"
    #     # @BUG
    #     # fast get static steps @fangjing --> don't use hash to dynamic sample
    #     steps_filename =  "steps_data_index.pkl"
    #     steps_filename = "steps_332420bad1ab.pkl"

    #     steps_path = self.dataset_path / "meta" / steps_filename
        
    #     # Try to load cached steps first
    #     try:
    #         if steps_path.exists():
    #             with open(steps_path, "rb") as f:
    #                 cached_data = pickle.load(f)
    #             return cached_data["steps"]
    #         else:
    #             steps_filename = "steps_2d5a34b904d2.pkl"
    #             steps_path = self.dataset_path / "meta" / steps_filename
        
    #             with open(steps_path, "rb") as f:
    #                 cached_data = pickle.load(f)
    #             return cached_data["steps"]


    #     except (FileNotFoundError, pickle.PickleError, KeyError) as e:
    #         print(f"Failed to load cached steps: {e}")
    #         print("Computing steps from scratch...")

    #     # Compute steps using single process
    #     all_steps = self._get_all_steps_single_process()
        
    #     # Cache the computed steps with unique filename
    #     try:
    #         cache_data = {
    #             "config_key": config_key,
    #             "steps": all_steps,
    #             "num_trajectories": len(self.trajectory_ids),
    #             "total_steps": len(all_steps),
    #             "computed_timestamp": pd.Timestamp.now().isoformat(),
    #             "delete_pause_frame": self.delete_pause_frame,
    #         }
            
    #         # Ensure the meta directory exists
    #         steps_path.parent.mkdir(parents=True, exist_ok=True)
            
    #         with open(steps_path, "wb") as f:
    #             pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    #         print(f"Cached steps saved to {steps_path}")
    #     except Exception as e:
    #         print(f"Failed to cache steps: {e}")
        
    #     return all_steps

    def _get_steps_config_key(self) -> str:
        """Generate a configuration key for steps caching."""
        config_dict = {
            "delete_pause_frame": self.delete_pause_frame,
            "dataset_name": self.dataset_name,
        }
        # Create a hash of the configuration
        config_str = str(sorted(config_dict.items()))
        return hashlib.md5(config_str.encode()).hexdigest()[:12]  #


    def _get_all_steps_single_process(self) -> list[tuple[int, int]]:
        """Original single-process implementation as fallback."""
        all_steps: list[tuple[int, int]] = []
        skipped_trajectories = 0
        processed_trajectories = 0
        
        # Check if language modality is configured
        has_language_modality = 'language' in self.modality_keys and len(self.modality_keys['language']) > 0
        
        for trajectory_id, trajectory_length in tqdm(zip(self.trajectory_ids, self.trajectory_lengths), total=len(self.trajectory_ids), desc="Getting All Step"):

            for base_index in range(trajectory_length):
                all_steps.append((trajectory_id, base_index))

            ## we do not check trajectory following GR00T
            # data = self.get_trajectory_data(trajectory_id)
            # trajectory_skipped = False
            
            # # Check if trajectory has valid language instruction (if language modality is configured)
            # if has_language_modality:
            #     self.curr_traj_data = data  # Set current trajectory data for get_language to work
            #     try:
            #         language_instruction = self.get_language(trajectory_id, self.modality_keys['language'][0], 0)
            #         if not language_instruction or language_instruction[0] == "":
            #             print(f"Skipping trajectory {trajectory_id} due to empty language instruction")
            #             skipped_trajectories += 1
            #             trajectory_skipped = True
            #             continue
            #     except Exception as e:
            #         print(f"Skipping trajectory {trajectory_id} due to language retrieval error: {e}")
            #         skipped_trajectories += 1
            #         trajectory_skipped = True
            #         continue
            
            # if not trajectory_skipped:
            #     processed_trajectories += 1
            
            # if self.delete_pause_frame:
            #     # Get position and gripper fields based on available columns
            #     delta_position_values, gripper_values = self._get_position_and_gripper_values(data)
            #     previous_gripper = gripper_values[0]
            #     for base_index in range(trajectory_length):
            #         if base_index >= len(delta_position_values) or base_index >= len(gripper_values):
            #             break
                        
            #         # Check for translation change using the detected position fields
            #         has_translation_change = np.any(np.abs(delta_position_values[base_index]) > EPSILON)
            #         has_gripper_change = gripper_values[base_index] != (previous_gripper if base_index == 0 else gripper_values[base_index-1])
                    
            #         if has_translation_change or has_gripper_change:
            #             all_steps.append((trajectory_id, base_index))
            # else:
            #     for base_index in range(trajectory_length):
            #         all_steps.append((trajectory_id, base_index))
                    
        # Print summary statistics
        print(f"Single-process summary: Processed {processed_trajectories} trajectories, skipped {skipped_trajectories} empty trajectories")
        print(f"Total steps: {len(all_steps)} from {len(self.trajectory_ids)} trajectories")
                   
        return all_steps

    def _get_position_and_gripper_values(self, data: pd.DataFrame) -> tuple[list, list]:
        """Get position and gripper values based on available columns in the dataset."""
        # Get action keys from modality_keys
        action_keys = self.modality_keys.get('action', [])
        
        # Extract position data
        delta_position_values = None
        position_candidates = ['delta_eef_position']
        coordinate_candidates = ['x', 'y', 'z']
        
        # First try combined position fields
        for pos_key in position_candidates:
            full_key = f"action.{pos_key}"
            if full_key in action_keys:
                try:
                    # Get the lerobot key for this modality
                    le_action_cfg = self.lerobot_modality_meta.action
                    subkey = pos_key
                    if subkey in le_action_cfg:
                        le_key = le_action_cfg[subkey].original_key or subkey
                        if le_key in data.columns:
                            data_array = np.stack(data[le_key])
                            le_indices = np.arange(le_action_cfg[subkey].start, le_action_cfg[subkey].end)
                            filtered_data = data_array[:, le_indices]
                            delta_position_values = filtered_data.tolist()
                            break
                except Exception:
                    continue
        
        # If combined fields not found, try individual x,y,z coordinates
        if delta_position_values is None:
            x_data, y_data, z_data = None, None, None
            for coord in coordinate_candidates:
                full_key = f"action.{coord}"
                if full_key in action_keys:
                    try:
                        le_action_cfg = self.lerobot_modality_meta.action
                        if coord in le_action_cfg:
                            le_key = le_action_cfg[coord].original_key or coord
                            if le_key in data.columns:
                                data_array = np.stack(data[le_key])
                                le_indices = np.arange(le_action_cfg[coord].start, le_action_cfg[coord].end)
                                coord_data = data_array[:, le_indices].flatten()
                                if coord == 'x':
                                    x_data = coord_data
                                elif coord == 'y':
                                    y_data = coord_data
                                elif coord == 'z':
                                    z_data = coord_data
                    except Exception:
                        continue
            
            if x_data is not None and y_data is not None and z_data is not None:
                delta_position_values = np.column_stack((x_data, y_data, z_data)).tolist()
        
        if delta_position_values is None:
            # Fallback to the old hardcoded approach if metadata approach fails
            if 'action.delta_eef_position' in data.columns:
                delta_position_values = data['action.delta_eef_position'].to_numpy().tolist()
            elif all(col in data.columns for col in ['action.x', 'action.y', 'action.z']):
                x_vals = data['action.x'].to_numpy()
                y_vals = data['action.y'].to_numpy() 
                z_vals = data['action.z'].to_numpy()
                delta_position_values = np.column_stack((x_vals, y_vals, z_vals)).tolist()
            else:
                raise ValueError(f"No suitable position columns found. Available columns: {data.columns.tolist()}")
        
        # Extract gripper data
        gripper_values = None
        gripper_candidates = ['gripper_close', 'gripper']
        
        for grip_key in gripper_candidates:
            full_key = f"action.{grip_key}"
            if full_key in action_keys:
                try:
                    le_action_cfg = self.lerobot_modality_meta.action
                    if grip_key in le_action_cfg:
                        le_key = le_action_cfg[grip_key].original_key or grip_key
                        if le_key in data.columns:
                            data_array = np.stack(data[le_key])
                            le_indices = np.arange(le_action_cfg[grip_key].start, le_action_cfg[grip_key].end)
                            gripper_data = data_array[:, le_indices].flatten()
                            gripper_values = gripper_data.tolist()
                            break
                except Exception:
                    continue
        
        if gripper_values is None:
            # Fallback to the old hardcoded approach if metadata approach fails
            if 'action.gripper_close' in data.columns:
                gripper_values = data['action.gripper_close'].to_numpy().tolist()
            elif 'action.gripper' in data.columns:
                gripper_values = data['action.gripper'].to_numpy().tolist()
            else:
                raise ValueError(f"No suitable gripper columns found. Available columns: {data.columns.tolist()}")
        
        return delta_position_values, gripper_values

    def _get_modality_keys(self) -> dict:
        """Get the modality keys for the dataset.
        The keys are the modality names, and the values are the keys for each modality.
        See property `modality_keys` for the expected format.
        """
        modality_keys = defaultdict(list)
        for modality, config in self.modality_configs.items():
            modality_keys[modality] = config.modality_keys
        return modality_keys

    def _get_delta_indices(self) -> dict[str, np.ndarray]:
        """Restructure the delta indices to use modality.key as keys instead of just the modalities."""
        delta_indices: dict[str, np.ndarray] = {}
        for config in self.modality_configs.values():
            for key in config.modality_keys:
                delta_indices[key] = np.array(config.delta_indices)
        return delta_indices

    def _get_lerobot_modality_meta(self) -> LeRobotModalityMetadata:
        """Get the metadata for the LeRobot dataset."""
        if self._modality_filename is not None:
            modality_meta_path = self.dataset_path / "meta" / self._modality_filename
        else:
            modality_meta_path = self.dataset_path / LE_ROBOT_MODALITY_FILENAME
        assert (
            modality_meta_path.exists()
        ), f"Please provide {modality_meta_path} (modality metadata) in {self.dataset_path}"
        with open(modality_meta_path, "r") as f:
            modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))
        return modality_meta

    def _get_lerobot_info_meta(self) -> dict:
        """Get the metadata for the LeRobot dataset."""
        if self._info_filename is not None:
            info_meta_path = self.dataset_path / "meta" / self._info_filename
        else:
            info_meta_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        assert info_meta_path.exists(), f"Please provide {info_meta_path} (info metadata) in {self.dataset_path}"
        with open(info_meta_path, "r") as f:
            info_meta = json.load(f)
        return info_meta

    def _get_data_path_pattern(self) -> str:
        """Get the data path pattern for the LeRobot dataset."""
        return self.lerobot_info_meta["data_path"]

    def _get_video_path_pattern(self) -> str:
        """Get the video path pattern for the LeRobot dataset."""
        return self.lerobot_info_meta["video_path"]

    def _get_chunk_size(self) -> int:
        """Get the chunk size for the LeRobot dataset."""
        return self.lerobot_info_meta["chunks_size"]

    def _get_tasks(self) -> pd.DataFrame:
        """Get the tasks for the dataset."""
        tasks_path = self.dataset_path / LE_ROBOT_TASKS_FILENAME
        with open(tasks_path, "r", encoding="utf-8") as f:
            tasks = [json.loads(line) for line in f]
        df = pd.DataFrame(tasks)
        return df.set_index("task_index")

    def _load_simple_tasks_mapping(self) -> dict[str, list[str]] | None:
        """Load task augmentation mapping with priority: tasks_rewritten > tasks_simple_v2.

        Called only when ``use_simple_tasks`` is True. Tries ``meta/tasks_rewritten.jsonl`` first,
        then ``meta/tasks_simple_v2.jsonl``, regardless of ``use_episode_instruction``: episode-level
        instructions still resolve originals from ``episodes.jsonl`` when enabled, and task-index
        datasets still resolve originals via parquet ``annotation.task_index`` + ``tasks.jsonl``.

        Both files share the same format:
        {"task_index": 0, "task": ["main task", "variation1", "variation2", ...]}

        Returns:
            dict[str, list[str]] | None: Mapping from main task string to list of all task variations,
                or None if no augmentation file is found at ``dataset_path``.
        """
        candidates = [
            LE_ROBOT_TASKS_REWRITTEN_FILENAME,
            LE_ROBOT_TASKS_SIMPLE_FILENAME,
        ]
        chosen_path = None
        for fname in candidates:
            p = self.dataset_path / fname
            if p.exists():
                chosen_path = p
                break

        if chosen_path is None:
            import warnings
            warnings.warn(
                f"use_simple_tasks is enabled but neither {LE_ROBOT_TASKS_REWRITTEN_FILENAME} nor "
                f"{LE_ROBOT_TASKS_SIMPLE_FILENAME} found at {self.dataset_path}. "
                f"Falling back to {LE_ROBOT_TASKS_FILENAME} (no task augmentation for this dataset).",
                stacklevel=2,
            )
            return None

        mapping: dict[str, list[str]] = {}
        with open(chosen_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                task_list = entry["task"]
                assert isinstance(task_list, list) and len(task_list) >= 1, (
                    f"Expected 'task' field to be a non-empty list, got {task_list} in {chosen_path}"
                )
                main_task = task_list[0]
                mapping[main_task] = task_list

        print(f"Loaded simple tasks mapping with {len(mapping)} entries from {chosen_path}")
        return mapping

    def _check_integrity(self):
        """Use the config to check if the keys are valid and detect silent data corruption."""
        ERROR_MSG_HEADER = f"Error occurred in initializing dataset {self.dataset_name}:\n"

        for modality_config in self.modality_configs.values():
            for key in modality_config.modality_keys:
                if key == "lapa_action" or key == "dream_actions":
                    continue  # no need for any metadata for lapa actions because it comes normalized
                # Check if the key is valid
                try:
                    self.lerobot_modality_meta.get_key_meta(key)
                except Exception as e:
                    raise ValueError(
                        ERROR_MSG_HEADER + f"Unable to find key {key} in modality metadata:\n{e}"
                    )

    def set_transforms_metadata(self, metadata: DatasetMetadata):
        """Set transform metadata while keeping this dataset's actual video geometry."""
        if metadata is not self.metadata and getattr(self.metadata.modalities, "video", None) is not None:
            import copy

            metadata = copy.deepcopy(metadata)
            metadata.modalities.video = self.metadata.modalities.video
        self.transforms.set_metadata(metadata)

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset.

        Args:
            epoch (int): The epoch to set.
        """
        self.epoch = epoch

    def __len__(self) -> int:
        """Get the total number of data points in the dataset.

        Returns:
            int: the total number of data points in the dataset.
        """
        return len(self.all_steps)

    def __str__(self) -> str:
        """Get the description of the dataset."""
        return f"{self.dataset_name} ({len(self)} steps)"

    ## do not use this getitem in LeRobotSingleDataset
    def __getitem__(self, index: int) -> dict:
        """Get the data for a single step in a trajectory.

        Args:
            index (int): The index of the step to get.

        Returns:
            dict: The data for the step.
        """
        trajectory_id, base_index = self.all_steps[index]
        
        data = self.get_step_data(trajectory_id, base_index)
        
        # Process all video keys dynamically
        images = []
        for video_key in self.modality_keys["video"]:
            image = data[video_key][0]
            
            # Apply image cropping if enabled and the video key is base_view
            # Note: crop_obs_camera functionality has been removed
            
            image = Image.fromarray(image).resize((224, 224))
            images.append(image)
        
        # Get language and action data
        language = data[self.modality_keys["language"][0]][0]
        action = []
        for action_key in self.modality_keys["action"]:
            action.append(data[action_key])
        action = np.concatenate(action, axis=1)
        
        state = []
        for state_key in self.modality_keys["state"]:
            state.append(data[state_key])
        state = np.concatenate(state, axis=1)

        return dict(action=action, state=state, image=images, language=language)

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        """Get the RAW data for a single step in a trajectory. No transforms are applied.

        Args:
            trajectory_id (int): The name of the trajectory.
            base_index (int): The base step index in the trajectory.

        Returns:
            dict: The RAW data for the step.

        Example return:
            {
                "video": {
                    "video.image_side_0": [B, T, H, W, C],
                    "video.image_side_1": [B, T, H, W, C],
                },
                "state": {
                    "state.eef_position": [B, T, state_dim],
                    "state.eef_rotation": [B, T, state_dim],
                },
                "action": {
                    "action.eef_position": [B, T, action_dim],
                    "action.eef_rotation": [B, T, action_dim],
                },
            }
        """
        data = {}
        # Get the data for all modalities
        self.curr_traj_data = self.get_trajectory_data(trajectory_id)
        # TODO @JinhuiYE The logic below is poorly implemented. Data reading should be directly based on curr_traj_data.
        for modality in self.modality_keys:
            # Get the data corresponding to each key in the modality
            for key in self.modality_keys[modality]:
                data[key] = self.get_data_by_modality(trajectory_id, modality, key, base_index)
        return data

    def get_trajectory_data(self, trajectory_id: int) -> pd.DataFrame:
        """Get the data for a trajectory."""
        if self.curr_traj_id == trajectory_id and self.curr_traj_data is not None:
            return self.curr_traj_data
        else:
            chunk_index = self.get_episode_chunk(trajectory_id)
            parquet_path = self.dataset_path / self.data_path_pattern.format(
                episode_chunk=chunk_index, episode_index=trajectory_id
            )
            assert parquet_path.exists(), f"Parquet file not found at {parquet_path}"
            return pd.read_parquet(parquet_path)

    def get_trajectory_index(self, trajectory_id: int) -> int:
        """Get the index of the trajectory in the dataset by the trajectory ID.
        This is useful when you need to get the trajectory length or sampling weight corresponding to the trajectory ID.

        Args:
            trajectory_id (str): The ID of the trajectory.

        Returns:
            int: The index of the trajectory in the dataset.
        """
        trajectory_indices = np.where(self.trajectory_ids == trajectory_id)[0]
        if len(trajectory_indices) != 1:
            raise ValueError(
                f"Error finding trajectory index for {trajectory_id}, found {trajectory_indices=}"
            )
        return trajectory_indices[0]

    def get_episode_chunk(self, ep_index: int) -> int:
        """Get the chunk index for an episode index."""
        return ep_index // self.chunk_size

    ## definitely should be first_last padding, not zero in the original code
    def retrieve_data_and_pad(
        self,
        array: np.ndarray,
        step_indices: np.ndarray,
        max_length: int,
        padding_strategy: str = "first_last",
    ) -> np.ndarray:
        """Retrieve the data from the dataset and pad it if necessary.
        Args:
            array (np.ndarray): The array to retrieve the data from.
            step_indices (np.ndarray): The step indices to retrieve the data for.
            max_length (int): The maximum length of the data.
            padding_strategy (str): The padding strategy, either "first" or "last".
        """
        # Get the padding indices
        front_padding_indices = step_indices < 0
        end_padding_indices = step_indices >= max_length
        padding_positions = np.logical_or(front_padding_indices, end_padding_indices)
        # Retrieve the data with the non-padding indices
        # If there exists some padding, Given T step_indices, the shape of the retrieved data will be (T', ...) where T' < T
        raw_data = array[step_indices[~padding_positions]]
        assert isinstance(raw_data, np.ndarray), f"{type(raw_data)=}"
        # This is the shape of the output, (T, ...)
        if raw_data.ndim == 1:
            expected_shape = (len(step_indices),)
        else:
            expected_shape = (len(step_indices), *array.shape[1:])

        # Pad the data
        output = np.zeros(expected_shape)
        # Assign the non-padded data
        output[~padding_positions] = raw_data
        # If there exists some padding, pad the data
        if padding_positions.any():
            #print(padding_positions, front_padding_indices, end_padding_indices, padding_strategy)
            if padding_strategy == "first_last":
                # Use first / last step data to pad
                front_padding_data = array[0]
                end_padding_data = array[-1]
                output[front_padding_indices] = front_padding_data
                output[end_padding_indices] = end_padding_data
            elif padding_strategy == "zero":
                # Use zero padding
                output[padding_positions] = 0
            else:
                raise ValueError(f"Invalid padding strategy: {padding_strategy}")
        return output

    def _interpolate_and_pad(
        self,
        array: np.ndarray,
        base_index: int,
        delta_indices: np.ndarray,
        max_length: int,
    ) -> np.ndarray:
        """Retrieve state/action data with linear interpolation for FPS resampling.

        Maps target-FPS delta_indices to fractional source-FPS positions, then
        linearly interpolates between adjacent source frames.  Out-of-range
        positions are clamped to boundary values (same semantics as first_last
        padding in retrieve_data_and_pad).

        Args:
            array: (T_src, D) full trajectory data at source FPS.
            base_index: current step index in the source trajectory.
            delta_indices: target-FPS offsets, e.g. [0, 1, ..., 49].
            max_length: source trajectory length (for boundary clamping).
        """
        target_times = delta_indices.astype(np.float64) / self.target_fps
        src_fractional = target_times * self.source_fps + base_index

        src_clamped = np.clip(src_fractional, 0, max_length - 1)

        src_floor = np.floor(src_clamped).astype(int)
        src_ceil = np.minimum(src_floor + 1, max_length - 1)
        weights = (src_clamped - src_floor)[:, None]  # (T_tgt, 1)

        return array[src_floor] * (1.0 - weights) + array[src_ceil] * weights

    def get_video_path(self, trajectory_id: int, key: str) -> Path:
        chunk_index = self.get_episode_chunk(trajectory_id)
        original_key = self.lerobot_modality_meta.video[key].original_key
        if original_key is None:
            original_key = key
        video_filename = self.video_path_pattern.format(
            episode_chunk=chunk_index, episode_index=trajectory_id, video_key=original_key
        )
        return self.dataset_path / video_filename

    def get_video(
        self,
        trajectory_id: int,
        key: str,
        base_index: int,
    ) -> np.ndarray:
        """Get the video frames for a trajectory by a base index.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (str): The ID of the trajectory.
            key (str): The key of the video.
            base_index (int): The base index of the trajectory.

        Returns:
            np.ndarray: The video frames for the trajectory and frame indices. Shape: (T, H, W, C)
        """
        # Get the step indices
        step_indices = self.delta_indices[key] + base_index
        # print(f"{step_indices=}")
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Ensure the indices are within the valid range
        # This is equivalent to padding the video with extra frames at the beginning and end
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, self.trajectory_lengths[trajectory_index] - 1)
        assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
        # Get the sub-key
        key = key.replace("video.", "")
        video_path = self.get_video_path(trajectory_id, key)
        # Get the action/state timestamps for each frame in the video
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert "timestamp" in self.curr_traj_data.columns, f"No timestamp found in {trajectory_id=}"
        timestamp: np.ndarray = self.curr_traj_data["timestamp"].to_numpy()
        # Get the corresponding video timestamps from the step indices
        video_timestamp = timestamp[step_indices]

        rgb_frame = get_frames_by_timestamps(
            video_path.as_posix(),
            video_timestamp,
            video_backend=self.video_backend,
            video_backend_kwargs=self.video_backend_kwargs,
        )

        return rgb_frame

    def get_state_or_action(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        base_index: int,
    ) -> np.ndarray:
        """Get the state or action data for a trajectory by a base index.
        If the step indices are out of range, pad with the data:
            if the data is stored in absolute format, pad with the first or last step data;
            otherwise, pad with zero.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            modality (str): The modality of the data.
            key (str): The key of the data.
            base_index (int): The base index of the trajectory.

        Returns:
            np.ndarray: The data for the trajectory and step indices.
        """
        full_key = key
        # Get the step indices
        step_indices = self.delta_indices[full_key] + base_index
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Get the maximum length of the trajectory
        max_length = self.trajectory_lengths[trajectory_index]
        assert key.startswith(modality + "."), f"{key} must start with {modality + '.'}, got {key}"
        # Get the sub-key, e.g. state.joint_angles -> joint_angles
        key = key.replace(modality + ".", "")
        # Get the lerobot key
        le_state_or_action_cfg = getattr(self.lerobot_modality_meta, modality)
        le_key = le_state_or_action_cfg[key].original_key
        if le_key is None:
            le_key = key
        # Get the data array, shape: (T, D)
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert le_key in self.curr_traj_data.columns, f"No {le_key} found in {trajectory_id=}"
        data_array: np.ndarray = np.stack(self.curr_traj_data[le_key])  # type: ignore
        if data_array.ndim == 1:
            data_array = data_array[:, np.newaxis]
        assert data_array.ndim == 2, f"Expected 2D array, got key {le_key} is{data_array.shape} array"
        le_indices = np.arange(
            le_state_or_action_cfg[key].start,
            le_state_or_action_cfg[key].end,
        )
        data_array = data_array[:, le_indices]
        # Get the state or action configuration
        state_or_action_cfg = getattr(self.metadata.modalities, modality)[key]

        if self.target_fps is not None and self.target_fps != self.source_fps:
            return self._interpolate_and_pad(
                array=data_array,
                base_index=base_index,
                delta_indices=self.delta_indices[full_key],
                max_length=max_length,
            )

        # Pad the data
        return self.retrieve_data_and_pad(
            array=data_array,
            step_indices=step_indices,
            max_length=max_length,
            padding_strategy="first_last",
        )

    def _sample_from_simple_tasks(self, task: str) -> str:
        """Sample a random task variation from simple_tasks_mapping.
        
        Args:
            task: Original task string to look up.
            
        Returns:
            A randomly selected task variation, or the original task if not found in mapping.
        """
        assert self._simple_tasks_mapping is not None, "simple_tasks_mapping is None, should not call this method"
        if task in self._simple_tasks_mapping:
            return random.choice(self._simple_tasks_mapping[task])
        else:
            # Task not found in mapping, keep original
            return task

    def get_language(
        self,
        trajectory_id: int,
        key: str,
        base_index: int,
    ) -> list[str]:
        """Get the language annotation data for a trajectory by step indices.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            key (str): The key of the annotation.
            base_index (int): The base index of the trajectory.

        Returns:
            list[str]: The annotation data for the trajectory and step indices. If no matching data is found, return empty strings.
        """
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        # Get the step indices
        step_indices = self.delta_indices[key] + base_index
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)

        if self.trajectory_tasks is not None:
            # Use trajectory_tasks from episodes.jsonl
            original_tasks = self.trajectory_tasks[trajectory_index]
            if self._simple_tasks_mapping is not None:
                tasks = [self._sample_from_simple_tasks(t) for t in original_tasks]
            else:
                tasks = original_tasks
            if self.strip_bilingual_task_prefix:
                tasks = [t.split("@", 1)[1] if "@" in t else t for t in tasks]
            return tasks

        # Fallback: use self.tasks from tasks.jsonl
        # Get the maximum length of the trajectory
        max_length = self.trajectory_lengths[trajectory_index]
        # Get the end times corresponding to the closest indices
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, max_length - 1)
        # Get the annotations
        task_indices: list[int] = []
        assert key.startswith(
            "annotation."
        ), f"Language key must start with 'annotation.', got {key}"
        subkey = key.replace("annotation.", "")
        annotation_meta = self.lerobot_modality_meta.annotation

        assert annotation_meta is not None, f"Annotation metadata is None for {subkey}"
        assert (
            subkey in annotation_meta
        ), f"Annotation key {subkey} not found in metadata, available annotation keys: {annotation_meta.keys()}"
        subkey_meta = annotation_meta[subkey]
        original_key = subkey_meta.original_key
        if original_key is None:
            original_key = key
            if original_key not in self.curr_traj_data.keys():
                original_key = subkey

        for i in range(len(step_indices)):
            task_indices.append(self.curr_traj_data[original_key][step_indices[i]].item())
        original_tasks = self.tasks.loc[task_indices]["task"].tolist()
        
        if self._simple_tasks_mapping is not None:
            tasks = [self._sample_from_simple_tasks(t) for t in original_tasks]
        else:
            tasks = original_tasks
        if self.strip_bilingual_task_prefix:
            tasks = [t.split("@", 1)[1] if "@" in t else t for t in tasks]
        return tasks

    def get_data_by_modality(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        base_index: int,
    ):
        """Get the data corresponding to the modality for a trajectory by a base index.
        This method will call the corresponding helper method based on the modality.
        See the helper methods for more details.
        NOTE: For the language modality, the data is padded with empty strings if no matching data is found.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            modality (str): The modality of the data.
            key (str): The key of the data.
            base_index (int): The base index of the trajectory.
        """
        if modality == "video":
            return self.get_video(trajectory_id, key, base_index)
        elif modality == "state" or modality == "action":
            return self.get_state_or_action(trajectory_id, modality, key, base_index)
        elif modality == "language":
            return self.get_language(trajectory_id, key, base_index)
        else:
            raise ValueError(f"Invalid modality: {modality}")

    def save_dataset_statistics(self, save_path: Path | str, format: str = "json") -> None:
        """
        Save dataset statistics to specified path in the required format.
        Only includes statistics for keys that are actually used in the dataset.
        Gripper-related keys will be placed at the end.
        
        Args:
            save_path (Path | str): Path to save the statistics file
            format (str): Save format, currently only supports "json"
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build the data structure to save
        statistics_data = {}
        
        # Get used modality keys
        used_action_keys, used_state_keys = get_used_modality_keys(self.modality_keys)
        
        # Organize statistics by tag
        tag = self.tag
        tag_stats = {}
        
        # Process action statistics (only for used keys)
        if hasattr(self.metadata.statistics, 'action') and self.metadata.statistics.action:
            action_stats = self.metadata.statistics.action
            
            # Filter to only include used action keys and reorder: non-gripper first, gripper last
            non_gripper_keys = []
            gripper_keys = []
            
            for key in action_stats.keys():
                if key in used_action_keys:
                    if "gripper" in key.lower():
                        gripper_keys.append(key)
                    else:
                        non_gripper_keys.append(key)
            
            # Reorder: non-gripper first, gripper last
            reordered_keys = non_gripper_keys + gripper_keys
            
            filtered_action_stats = {}
            for key in reordered_keys:
                filtered_action_stats[key] = action_stats[key]
            
            if filtered_action_stats:
                # Combine statistics from filtered action sub-keys
                combined_action_stats = combine_modality_stats(filtered_action_stats)
                
                # Add mask field based on whether it's gripper or not
                mask = generate_action_mask_for_used_keys(
                    self.metadata.modalities.action, filtered_action_stats.keys()
                )
                combined_action_stats["mask"] = mask
                
                tag_stats["action"] = combined_action_stats
        
        # Process state statistics (only for used keys)
        if hasattr(self.metadata.statistics, 'state') and self.metadata.statistics.state:
            state_stats = self.metadata.statistics.state
            
            # Filter to only include used state keys, optionally reorder gripper to end
            non_gripper_keys = []
            gripper_keys = []
            
            for key in state_stats.keys():
                if key in used_state_keys:
                    if "gripper" in key.lower():
                        gripper_keys.append(key)
                    else:
                        non_gripper_keys.append(key)
            
            # Reorder: non-gripper first, gripper last
            reordered_keys = non_gripper_keys + gripper_keys
            
            filtered_state_stats = {}
            for key in reordered_keys:
                filtered_state_stats[key] = state_stats[key]
            
            if filtered_state_stats:
                combined_state_stats = combine_modality_stats(filtered_state_stats)
                tag_stats["state"] = combined_state_stats
        
        # Add dataset counts
        tag_stats["num_transitions"] = len(self)
        tag_stats["num_trajectories"] = len(self.trajectory_ids)
        
        statistics_data[tag] = tag_stats
        
        # Save as JSON file
        if format.lower() == "json":
            if not str(save_path).endswith('.json'):
                save_path = save_path.with_suffix('.json')
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(statistics_data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}. Currently only 'json' is supported.")
        
        print(f"Single dataset statistics saved to: {save_path}")
        print(f"Used action keys (reordered): {list(used_action_keys)}")
        print(f"Used state keys (reordered): {list(used_state_keys)}")


class CachedLeRobotSingleDataset(LeRobotSingleDataset):
    def __init__(self, img_resize: tuple[int, int] | None = None, *args, **kwargs):
        """
        This class caches the video frames for each trajectory and key.
        It is recommended to use this class if the video frames need to be accessed multiple times.

        Args:
            resize_img (tuple[int, int], optional): The size to resize the video frames to reduce memory usage.
        """
        # Convert img_resize to tuple if it is not already
        if img_resize is not None and not isinstance(img_resize, tuple):
            img_resize = tuple(img_resize)
            assert len(img_resize) == 2, f"Expected tuple of length 2, got {img_resize}"
        self.img_resize = img_resize

        # Initialize img_resize attribute first to ensure it exists
        super().__init__(*args, **kwargs)
        cached_frames: dict[str, np.ndarray] = {}

        for key in self.modality_keys["video"]:
            all_frames = []
            original_key = key
            key = key.replace("video.", "")
            for trajectory_id, trajectory_length in tqdm(
                zip(self.trajectory_ids, self.trajectory_lengths),
                total=len(self.trajectory_ids),
                desc=f"Caching {key} frames",
            ):
                video_path = self.get_video_path(trajectory_id, key)
                frames = get_all_frames(
                    video_path.as_posix(),
                    video_backend=self.video_backend,
                    video_backend_kwargs=self.video_backend_kwargs,
                    resize_size=img_resize,
                )
                assert frames.ndim == 4, f"Expected 4D array, got {frames.shape} array"
                assert frames.shape[3] == 3, f"Expected 3 channels, got {frames.shape[3]} channels"
                
                # Apply image cropping if enabled and the video key is base_view
                # Note: crop_obs_camera functionality has been removed
                
                # assert (
                #     frames.shape[0] == trajectory_length
                # ), f"Expected {trajectory_length} frames, got {frames.shape[0]} frames"
                all_frames.append(frames)
            cached_frames[key] = np.concatenate(all_frames, axis=0)
            print(f"{key}: {cached_frames[key].shape}")
        self.cached_frames = cached_frames
        self.start_indices = np.cumsum(self.trajectory_lengths) - self.trajectory_lengths

    def get_video(self, trajectory_id: int, key: str, base_index: int) -> np.ndarray:
        step_indices = self.delta_indices[key] + base_index
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Ensure the indices are within the valid range
        # This is equivalent to padding the video with extra frames at the beginning and end
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, self.trajectory_lengths[trajectory_index] - 1)
        assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
        # Get the sub-key
        key = key.replace("video.", "")
        # Calculate the absolute indices
        absolute_indices = self.start_indices[trajectory_index] + step_indices
        return self.cached_frames[key][absolute_indices]

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        """Get the RAW data for a single step. No transforms are applied.

        Args:
            trajectory_id (str): The ID of the trajectory.
            base_index (int): The base index of the step.

        Returns:
            dict: The data for the step.
        """
        data = {}
        self.curr_traj_data = self.get_trajectory_data(trajectory_id)
        # Get the data for all modalities
        for modality in self.modality_keys:
            # Get the data corresponding to each key in the modality
            for key in self.modality_keys[modality]:
                data[key] = self.get_data_by_modality(trajectory_id, modality, key, base_index)
        return data

    def set_transforms_metadata(self, metadata: DatasetMetadata):
        """Set the metadata for the transforms. This is useful for transforms that need to know the metadata, such as the normalization values."""
        if self.img_resize is not None:
            all_video_keys = [key for key in self.modality_keys["video"]]
            for key in metadata.modalities.video:
                if key in all_video_keys:
                    metadata.modalities.video[key].resolution = self.img_resize
        super().set_transforms_metadata(metadata)


def safe_hash(input_tuple):
    # keep 128 bits of the hash
    tuple_string = repr(input_tuple).encode("utf-8")
    sha256 = hashlib.sha256()
    sha256.update(tuple_string)

    seed = int(sha256.hexdigest(), 16)

    return seed & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF


class MixtureSpecElement(BaseModel):
    dataset_path: list[Path] | Path = Field(..., description="The path to the dataset.")
    dataset_weight: float = Field(..., description="The weight of the dataset in the mixture.")
    distribute_weights: bool = Field(
        default=False,
        description="Whether to distribute the weights of the dataset across all the paths. If True, the weights will be evenly distributed across all the paths.",
    )


# Helper functions for dataset statistics

def combine_modality_stats(modality_stats: dict) -> dict:
    """
    Combine statistics from all sub-keys under a modality.
    
    Args:
        modality_stats (dict): Statistics for a modality, containing multiple sub-keys.
                               Each sub-key contains DatasetStatisticalValues object.
        
    Returns:
        dict: Combined statistics
    """
    combined_stats = {
        "mean": [],
        "std": [],
        "max": [],
        "min": [],
        "q01": [],
        "q99": []
    }
    
    # Combine statistics in sub-key order
    for subkey in modality_stats.keys():
        subkey_stats = modality_stats[subkey]  # This is a DatasetStatisticalValues object
        
        # Convert DatasetStatisticalValues to dict-like access
        for stat_name in ["mean", "std", "max", "min", "q01", "q99"]:
            stat_value = getattr(subkey_stats, stat_name)
            if isinstance(stat_value, (list, tuple)):
                combined_stats[stat_name].extend(stat_value)
            else:
                # Handle NDArray case - convert to list
                if hasattr(stat_value, 'tolist'):
                    combined_stats[stat_name].extend(stat_value.tolist())
                else:
                    combined_stats[stat_name].append(float(stat_value))
    
    return combined_stats

def generate_action_mask_for_used_keys(action_modalities: dict, used_action_keys_ordered) -> list[bool]:
    """
    Generate mask based on action modalities, but only for used keys.
    Gripper-related are False, others are True.
    
    Args:
        action_modalities (dict): Configuration information for action modalities.
        used_action_keys_ordered: Iterable of actually used action keys in the correct order.
        
    Returns:
        list[bool]: List of mask values
    """
    mask = []
    
    # Generate mask in the same order as the statistics were combined
    for subkey in used_action_keys_ordered:
        if subkey in action_modalities:
            subkey_config = action_modalities[subkey]
            
            # Get dimension count from shape
            if hasattr(subkey_config, 'shape') and len(subkey_config.shape) > 0:
                dim_count = subkey_config.shape[0]
            else:
                dim_count = 1
            
            # Check if it's gripper-related
            is_gripper = "gripper" in subkey.lower()
            
            # Generate mask value for each dimension
            for _ in range(dim_count):
                mask.append(not is_gripper)  # gripper is False, others are True
    
    return mask

def get_used_modality_keys(modality_keys: dict) -> tuple[set, set]:
    """Extract used action and state keys from modality configuration."""
    used_action_keys = set()
    used_state_keys = set()
    
    # Extract action keys (remove "action." prefix)
    for action_key in modality_keys.get("action", []):
        if action_key.startswith("action."):
            clean_key = action_key.replace("action.", "")
            used_action_keys.add(clean_key)
    
    # Extract state keys (remove "state." prefix)  
    for state_key in modality_keys.get("state", []):
        if state_key.startswith("state."):
            clean_key = state_key.replace("state.", "")
            used_state_keys.add(clean_key)
    
    return used_action_keys, used_state_keys

class LeRobotMixtureDataset(Dataset):
    """
    A mixture of multiple datasets. This class samples a single dataset based on the dataset weights and then calls the `__getitem__` method of the sampled dataset.
    It is recommended to modify the single dataset class instead of this class.
    """

    def __init__(
        self,
        data_mixture: Sequence[tuple[LeRobotSingleDataset, float]],
        mode: str,
        balance_dataset_weights: bool = True,
        balance_trajectory_weights: bool = True,
        metadata_config: dict = {
            "percentile_mixing_method": "weighted_average",
        },
        image_size: tuple = (224, 224),
        use_delta: bool = False,
        dynamic_image_size: bool = False,
        seed: int = 42,
        cached_statistics_path: Path | str | None = None,
        disable_separate_embodiment_norm: bool = False,
        max_state_dim: int | None = None,
        max_action_dim: int | None = None,
    ):
        """
        Initialize the mixture dataset.

        Args:
            data_mixture (list[tuple[LeRobotSingleDataset, float]]): Datasets and their corresponding weights.
            mode (str): If "train", __getitem__ will return different samples every epoch; if "val" or "test", __getitem__ will return the same sample every epoch.
            balance_dataset_weights (bool): If True, the weight of dataset will be multiplied by the total trajectory length of each dataset.
            balance_trajectory_weights (bool): If True, sample trajectories within a dataset weighted by their length; otherwise, use equal weighting.
            dynamic_image_size (bool): If True, skip the forced .resize(image_size) in __getitem__ and let
                each data config's transforms determine the final image size. The Qwen processor then packs
                variable-size images with proper attention masks.
            seed (int): Random seed for sampling.
            disable_separate_embodiment_norm (bool): If True, merge normalization statistics across all
                datasets globally instead of grouping by embodiment tag. All embodiments will share the
                same normalization stats. Requires all datasets to have compatible stat keys.
        """
        datasets: list[LeRobotSingleDataset] = []
        dataset_sampling_weights: list[float] = []
        for dataset, weight in data_mixture:
            # Check if dataset is valid and has data
            if len(dataset) == 0:
                print(f"Warning: Skipping empty dataset {dataset.dataset_name}")
                continue
            datasets.append(dataset)
            dataset_sampling_weights.append(weight)
        
        if len(datasets) == 0:
            raise ValueError("No valid datasets found in the mixture. All datasets are empty.")
        
        self.datasets = datasets
        self.balance_dataset_weights = balance_dataset_weights
        self.balance_trajectory_weights = balance_trajectory_weights
        self.image_size = image_size
        self.use_delta = use_delta
        self.dynamic_image_size = dynamic_image_size
        self.seed = seed
        self.mode = mode
        self.disable_separate_embodiment_norm = disable_separate_embodiment_norm
        if cached_statistics_path is not None:
            self.cached_statistics_path = Path(cached_statistics_path)
        else:
            self.cached_statistics_path = None

        # Set properties for sampling

        # 1. Dataset lengths
        self._dataset_lengths = np.array([len(dataset) for dataset in self.datasets])
        print(f"Dataset lengths: {self._dataset_lengths}")

        # 2. Dataset sampling weights
        self._dataset_sampling_weights = np.array(dataset_sampling_weights)
        
        if self.balance_dataset_weights:
            self._dataset_sampling_weights *= self._dataset_lengths
        
        # Check for zero or negative weights before normalization
        if np.any(self._dataset_sampling_weights <= 0):
            print(f"Warning: Found zero or negative sampling weights: {self._dataset_sampling_weights}")
            # Set minimum weight to prevent division issues
            self._dataset_sampling_weights = np.maximum(self._dataset_sampling_weights, 1e-8)
        
        # Normalize weights
        weights_sum = self._dataset_sampling_weights.sum()
        if weights_sum == 0 or np.isnan(weights_sum):
            print(f"Error: Invalid weights sum: {weights_sum}")
            # Fallback to equal weights
            self._dataset_sampling_weights = np.ones(len(self.datasets)) / len(self.datasets)
            print(f"Fallback to equal weights")
        else:
            self._dataset_sampling_weights /= weights_sum

        # 3. Trajectory sampling weights
        self._trajectory_sampling_weights: list[np.ndarray] = []
        for i, dataset in enumerate(self.datasets):
            trajectory_sampling_weights = np.ones(len(dataset.trajectory_lengths))
            if self.balance_trajectory_weights:
                trajectory_sampling_weights *= dataset.trajectory_lengths
            
            # Check for zero or negative weights before normalization
            if np.any(trajectory_sampling_weights <= 0):
                print(f"Warning: Dataset {i} has zero or negative trajectory weights")
                trajectory_sampling_weights = np.maximum(trajectory_sampling_weights, 1e-8)
            
            # Normalize weights
            weights_sum = trajectory_sampling_weights.sum()
            if weights_sum == 0 or np.isnan(weights_sum):
                print(f"Error: Dataset {i} has invalid trajectory weights sum: {weights_sum}")
                # Fallback to equal weights
                trajectory_sampling_weights = np.ones(len(dataset.trajectory_lengths)) / len(dataset.trajectory_lengths)
            else:
                trajectory_sampling_weights /= weights_sum
            
            self._trajectory_sampling_weights.append(trajectory_sampling_weights)

        # 4. Primary dataset indices
        self._primary_dataset_indices = np.array(dataset_sampling_weights) == 1.0
        if not np.any(self._primary_dataset_indices):
            print(f"Warning: No dataset with weight 1.0 found. Original weights: {dataset_sampling_weights}")
            # Fallback: use the dataset(s) with maximum weight as primary
            max_weight = max(dataset_sampling_weights)
            self._primary_dataset_indices = np.array(dataset_sampling_weights) == max_weight
            print(f"Using datasets with maximum weight {max_weight} as primary: {self._primary_dataset_indices}")
            
        if not np.any(self._primary_dataset_indices):
            # This should never happen, but just in case
            print("Error: Still no primary dataset found. Using first dataset as primary.")
            self._primary_dataset_indices = np.zeros(len(self.datasets), dtype=bool)
            self._primary_dataset_indices[0] = True

        # Set the epoch and sample the first epoch
        self.set_epoch(0)

        self.update_metadata(metadata_config, cached_statistics_path=self.cached_statistics_path)

        # Compute max dims for padding (multi-embodiment mixing).
        configured_max_state_dim = max_state_dim
        configured_max_action_dim = max_action_dim
        computed_max_state_dim = 0
        computed_max_action_dim = 0
        for dataset in self.datasets:
            state_dim = self._compute_flattened_dim(dataset, modality="state")
            action_dim = self._compute_flattened_dim(dataset, modality="action")
            computed_max_state_dim = max(computed_max_state_dim, state_dim)
            computed_max_action_dim = max(computed_max_action_dim, action_dim)

        if configured_max_state_dim is not None:
            computed_max_state_dim = max(computed_max_state_dim, int(configured_max_state_dim))
        if configured_max_action_dim is not None:
            computed_max_action_dim = max(computed_max_action_dim, int(configured_max_action_dim))

        self.max_state_dim = int(computed_max_state_dim)
        self.max_action_dim = int(computed_max_action_dim)
        print(f"[LeRobotMixtureDataset] max_state_dim={self.max_state_dim}, max_action_dim={self.max_action_dim}")

        # Update each dataset's StateActionMaskTransform with global max dims
        # This allows the transform to handle all padding and mask generation
        # This is required because otherwise we need to specify max dims in data_config.py
        # Automatically updating makes it easier.
        self._update_transforms_max_dims()

    @property
    def dataset_lengths(self) -> np.ndarray:
        """The lengths of each dataset."""
        return self._dataset_lengths

    @property
    def dataset_sampling_weights(self) -> np.ndarray:
        """The sampling weights for each dataset."""
        return self._dataset_sampling_weights

    @property
    def trajectory_sampling_weights(self) -> list[np.ndarray]:
        """The sampling weights for each trajectory in each dataset."""
        return self._trajectory_sampling_weights

    @property
    def primary_dataset_indices(self) -> np.ndarray:
        """The indices of the primary datasets."""
        return self._primary_dataset_indices

    def __str__(self) -> str:
        dataset_descriptions = []
        for dataset, weight in zip(self.datasets, self.dataset_sampling_weights):
            dataset_description = {
                "Dataset": str(dataset),
                "Sampling weight": float(weight),
            }
            dataset_descriptions.append(dataset_description)
        return json.dumps({"Mixture dataset": dataset_descriptions}, indent=2)

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset.

        Args:
            epoch (int): The epoch to set.
        """
        self.epoch = epoch
        # self.sampled_steps = self.sample_epoch()

    def sample_step(self, index: int) -> tuple[LeRobotSingleDataset, int, int]:
        """Sample a single step from the dataset."""
        # return self.sampled_steps[index]

        # Set seed
        seed = index if self.mode != "train" else safe_hash((self.epoch, index, self.seed))
        rng = np.random.default_rng(seed)

        # Sample dataset
        dataset_index = rng.choice(len(self.datasets), p=self.dataset_sampling_weights)
        dataset = self.datasets[dataset_index]

        # Sample trajectory
        trajectory_index = rng.choice(
            len(dataset.trajectory_ids), p=self.trajectory_sampling_weights[dataset_index]
        )
        trajectory_id = dataset.trajectory_ids[trajectory_index]

        # Sample step
        base_index = rng.choice(dataset.trajectory_lengths[trajectory_index])
        return dataset, trajectory_id, base_index

    @staticmethod
    def _compute_flattened_dim(dataset: LeRobotSingleDataset, modality: str) -> int:
        """
        Compute flattened output dim for a modality.

        - For `state`, keys that go through StateActionSinCosTransform are doubled.
        """
        assert modality in {"state", "action"}

        # Collect sincos keys (only affects state).
        sincos_keys: set[str] = set()
        if modality == "state":
            try:
                from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionSinCosTransform

                for t in getattr(dataset.transforms, "transforms", []):
                    if isinstance(t, StateActionSinCosTransform):
                        sincos_keys.update(t.apply_to)
            except Exception:
                sincos_keys = set()

        keys = dataset.modality_keys.get(modality, [])
        if not keys:
            return 0

        modality_meta = getattr(dataset.metadata.modalities, modality)
        dim_total = 0

        for full_key in keys:
            _, subkey = full_key.split(".", 1)
            key_meta = modality_meta[subkey]
            key_dim = int(key_meta.shape[0])
            if modality == "state" and full_key in sincos_keys:
                key_dim *= 2
            dim_total += key_dim

        return dim_total

    def _update_transforms_max_dims(self) -> None:
        """
        Update each dataset's StateActionMaskTransform with the global max dims.
        This allows the transform to handle all padding and mask generation.
        """
        from starVLA.dataloader.gr00t_lerobot.transform.mask import StateActionMaskTransform

        for dataset in self.datasets:
            for transform in getattr(dataset.transforms, "transforms", []):
                if isinstance(transform, StateActionMaskTransform):
                    transform.max_state_dim = self.max_state_dim
                    transform.max_action_dim = self.max_action_dim
                    break

    @staticmethod
    def _pad_last_dim(x: np.ndarray, target_dim: int) -> np.ndarray:
        """Pad the last dimension of x to target_dim with zeros."""
        if x.shape[-1] == target_dim:
            return x
        if x.shape[-1] > target_dim:
            raise ValueError(f"Cannot pad: x_dim={x.shape[-1]} > target_dim={target_dim}")
        pad_width = [(0, 0)] * x.ndim
        pad_width[-1] = (0, target_dim - x.shape[-1])
        return np.pad(x, pad_width, mode="constant", constant_values=0)


    def __getitem__(self, index: int) -> dict:
        """Get the data for a single trajectory and start index.

        Args:
            index (int): The index of the trajectory to get.

        Returns:
            dict: The data for the trajectory and start index with keys:
                - state: [T, max_state_dim] padded state tensor
                - action: [T, max_action_dim] padded action tensor  
                - state_mask: [T, max_state_dim] mask for valid state dimensions
                - action_mask: [T, max_action_dim] mask for valid action dimensions
                - image: list of PIL images
                - lang: language instruction string

        Note: Requires transforms to include ConcatTransform and StateActionMaskTransform.
        """
        max_retries = 100
        last_exception = None
        keep_dataloader_gc = os.environ.get("STARVLA_KEEP_DATALOADER_GC", "0") == "1"
        if not keep_dataloader_gc and gc.isenabled():
            gc.disable()
        gc_collect_interval = 0 if keep_dataloader_gc else int(os.environ.get("STARVLA_DATALOADER_GC_COLLECT_INTERVAL", "100") or 0)
        if gc_collect_interval > 0:
            self._gc_collect_counter = getattr(self, "_gc_collect_counter", 0) + 1
        
        for attempt in range(max_retries):
            try:
                dataset, trajectory_name, step = self.sample_step(index)

                data_unnorm = dataset.get_step_data(trajectory_name, step)

                # if self.use_delta:
                #     for key in data_unnorm.keys():
                #         if 'action.' in key:
                #             data_unnorm[key] = data_unnorm[key] - data_unnorm[key.replace("action.", "state.")]

                ##dataset.transforms是in-place操作
                import copy
                import torch
                #data = dataset.transforms(data_unnorm)
                data_input_to_transform = copy.deepcopy(data_unnorm)
                # Dataset transforms are preprocessing only; keep them out of autograd
                # even when dataloader workers inherit PyTorch's default grad mode.
                with torch.no_grad():
                    data = dataset.transforms(data_input_to_transform)

                images = []
                for video_key in dataset.modality_keys["video"]:
                    for img_indx in range(data[video_key].shape[0]):
                        image = data[video_key][img_indx]
                        if self.dynamic_image_size:
                            image = Image.fromarray(image)
                        else:
                            image = Image.fromarray(image).resize(self.image_size)
                        images.append(image)

                # Get language and action data
                language = data[dataset.modality_keys["language"][0]][0]

                # Get state/action/masks from transforms (ConcatTransform + StateActionMaskTransform)
                # StateActionMaskTransform handles padding to max dims and mask generation
                assert "state" in data and "action" in data, \
                    "Transforms must include ConcatTransform. Missing 'state' or 'action' in transformed data."
                assert "action_mask" in data and "state_mask" in data, \
                    "Transforms must include StateActionMaskTransform. Missing 'action_mask' or 'state_mask' in transformed data."
                
                state = data["state"]
                action = data["action"]
                action_mask = data["action_mask"]
                state_mask = data["state_mask"]
                
                # ConcatTransform uses torch.cat, convert to numpy if needed
                if hasattr(state, 'numpy'):
                    state = state.detach().cpu().numpy().copy() if hasattr(state, "detach") else state.numpy().copy()
                if hasattr(action, 'numpy'):
                    action = action.detach().cpu().numpy().copy() if hasattr(action, "detach") else action.numpy().copy()

                # Concatenate and pad unnormalized action manually (transforms don't process unnorm data)
                action_unnorm = []
                for action_key in dataset.modality_keys["action"]:
                    action_unnorm.append(data_unnorm[action_key])
                action_unnorm = np.concatenate(action_unnorm, axis=1)
                action_unnorm = self._pad_last_dim(action_unnorm, self.max_action_dim)
                
                result = dict(
                    action=action,
                    action_unnorm=action_unnorm,
                    action_mask=action_mask,
                    state=state,
                    state_mask=state_mask,
                    image=images,
                    lang=language,
                    embodiment_tag=dataset.tag,
                )
                if gc_collect_interval > 0 and self._gc_collect_counter % gc_collect_interval == 0:
                    gc.collect()
                return result
                
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    suppress_retry_warning = os.getenv("IGNORE_DATALOADER_RETRY_WARNING") == "True"
                    if not suppress_retry_warning:
                        import traceback
                        traceback.print_exc()
                        # Log the error but continue trying
                        print(f"Attempt {attempt + 1}/{max_retries} failed for index {index}: {e}")
                        print(f"Retrying with new sample...")
                    # For retry, we can use a slightly different index to get a new sample
                    # This helps avoid getting stuck on the same problematic sample
                    index = (index + 1) % len(self)
                else:
                    # All retries exhausted
                    print(f"All {max_retries} attempts failed for index {index}")
                    print(f"Last error: {last_exception}")
                    # Return a dummy sample or re-raise the exception
                    raise last_exception

    def __len__(self) -> int:
        """Get the length of a single epoch in the mixture.

        Returns:
            int: The length of a single epoch in the mixture.
        """
        # Check for potential issues
        if len(self.datasets) == 0:
            return 0
            
        # Check if any dataset lengths are 0 or NaN
        if np.any(self.dataset_lengths == 0) or np.any(np.isnan(self.dataset_lengths)):
            print(f"Warning: Found zero or NaN dataset lengths: {self.dataset_lengths}")
            # Filter out zero/NaN length datasets
            valid_indices = (self.dataset_lengths > 0) & (~np.isnan(self.dataset_lengths))
            if not np.any(valid_indices):
                print("Error: All datasets have zero or NaN length")
                return 0
        else:
            valid_indices = np.ones(len(self.datasets), dtype=bool)
        
        # Check if any sampling weights are 0 or NaN
        if np.any(self.dataset_sampling_weights == 0) or np.any(np.isnan(self.dataset_sampling_weights)):
            print(f"Warning: Found zero or NaN sampling weights: {self.dataset_sampling_weights}")
            # Use only valid weights
            valid_weights = (self.dataset_sampling_weights > 0) & (~np.isnan(self.dataset_sampling_weights))
            valid_indices = valid_indices & valid_weights
            if not np.any(valid_indices):
                print("Error: All sampling weights are zero or NaN")
                return 0
        
        # Check primary dataset indices
        primary_and_valid = self.primary_dataset_indices & valid_indices
        if not np.any(primary_and_valid):
            print(f"Warning: No valid primary datasets found. Primary indices: {self.primary_dataset_indices}, Valid indices: {valid_indices}")
            # Fallback: use the largest valid dataset
            if np.any(valid_indices):
                max_length = self.dataset_lengths[valid_indices].max()
                print(f"Fallback: Using maximum dataset length: {max_length}")
                return int(max_length)
            else:
                return 0
        
        # Calculate the ratio and get max
        ratios = (self.dataset_lengths / self.dataset_sampling_weights)[primary_and_valid]
        
        # Check for NaN or inf in ratios
        if np.any(np.isnan(ratios)) or np.any(np.isinf(ratios)):
            print(f"Warning: Found NaN or inf in ratios: {ratios}")
            print(f"Dataset lengths: {self.dataset_lengths[primary_and_valid]}")
            print(f"Sampling weights: {self.dataset_sampling_weights[primary_and_valid]}")
            # Filter out invalid ratios
            valid_ratios = ratios[~np.isnan(ratios) & ~np.isinf(ratios)]
            if len(valid_ratios) == 0:
                print("Error: All ratios are NaN or inf")
                return 0
            max_ratio = valid_ratios.max()
        else:
            max_ratio = ratios.max()
        
        result = int(max_ratio)
        if result == 0:
            print(f"Warning: Dataset mixture length is 0")
        return result

    @staticmethod
    def compute_overall_statistics(
        per_task_stats: list[dict[str, dict[str, list[float] | np.ndarray]]],
        dataset_sampling_weights: list[float] | np.ndarray,
        percentile_mixing_method: str = "weighted_average",
    ) -> dict[str, dict[str, list[float]]]:
        """
        Computes overall statistics from per-task statistics using dataset sample weights.

        Args:
            per_task_stats: List of per-task statistics.
            Example format of one element in the per-task statistics list:
                {
                    "state.gripper": {
                        "min": [...],
                        "max": [...],
                        "mean": [...],
                        "std": [...],
                        "q01": [...],
                        "q99": [...],
                    },
                    ...
                }
            dataset_sampling_weights: List of sample weights for each task.
            percentile_mixing_method: The method to mix the percentiles, either "weighted_average" or "weighted_std".

        Returns:
            A dict of overall statistics per modality.
        """
        if len(per_task_stats) == 0:
            raise ValueError("per_task_stats must be non-empty")
        if len(per_task_stats) != len(dataset_sampling_weights):
            raise ValueError(
                "per_task_stats and dataset_sampling_weights must have the same length "
                f"(got {len(per_task_stats)} vs {len(dataset_sampling_weights)})"
            )
        # Normalize the sample weights to sum to 1
        dataset_sampling_weights = np.array(dataset_sampling_weights)
        normalized_weights = dataset_sampling_weights / dataset_sampling_weights.sum()

        # Initialize overall statistics dict
        overall_stats: dict[str, dict[str, list]] = {}

        # Get the list of modality keys
        modality_keys = per_task_stats[0].keys()

        for modality in modality_keys:
            # Detect if stats are 1D [dim] or 2D [chunk_size, dim]
            first_mean = np.array(per_task_stats[0][modality]["mean"])
            is_2d = first_mean.ndim == 2
            
            if is_2d:
                # 2D per-position stats: shape [chunk_size, dim]
                stats_shape = first_mean.shape
            else:
                # 1D stats: shape [dim]
                stats_shape = first_mean.shape

            # Initialize accumulators for means and variances
            weighted_means = np.zeros(stats_shape)
            weighted_squares = np.zeros(stats_shape)

            # Collect min, max, q01, q99 from all tasks
            min_list = []
            max_list = []
            q01_list = []
            q99_list = []

            for task_idx, task_stats in enumerate(per_task_stats):
                w_i = normalized_weights[task_idx]
                stats = task_stats[modality]
                means = np.array(stats["mean"])
                stds = np.array(stats["std"])

                # Update weighted sums for mean and variance
                weighted_means += w_i * means
                weighted_squares += w_i * (stds**2 + means**2)

                # Collect min, max, q01, q99
                min_list.append(stats["min"])
                max_list.append(stats["max"])
                q01_list.append(stats["q01"])
                q99_list.append(stats["q99"])

            # Compute overall mean
            overall_mean = weighted_means.tolist()

            # Compute overall variance and std deviation
            overall_variance = weighted_squares - weighted_means**2
            overall_std = np.sqrt(np.maximum(overall_variance, 0)).tolist()  # Clip to avoid sqrt of negative

            # Compute overall min and max per dimension
            # For 2D stats, axis=0 aggregates across tasks, preserving [chunk_size, dim]
            overall_min = np.min(np.array(min_list), axis=0).tolist()
            overall_max = np.max(np.array(max_list), axis=0).tolist()

            # Compute overall q01 and q99 per dimension
            # Use weighted average of per-task quantiles
            q01_array = np.array(q01_list)
            q99_array = np.array(q99_list)

            if percentile_mixing_method == "weighted_average":
                # For 2D stats, we need to broadcast weights correctly
                if is_2d:
                    # weights shape: [num_tasks] -> [num_tasks, 1, 1] for broadcasting
                    weights_bc = np.array(normalized_weights).reshape(-1, 1, 1)
                    weighted_q01 = (q01_array * weights_bc).sum(axis=0).tolist()
                    weighted_q99 = (q99_array * weights_bc).sum(axis=0).tolist()
                else:
                    weighted_q01 = np.average(q01_array, axis=0, weights=normalized_weights).tolist()
                    weighted_q99 = np.average(q99_array, axis=0, weights=normalized_weights).tolist()
            elif percentile_mixing_method == "min_max":
                weighted_q01 = np.min(q01_array, axis=0).tolist()
                weighted_q99 = np.max(q99_array, axis=0).tolist()
            else:
                raise ValueError(f"Invalid percentile mixing method: {percentile_mixing_method}")

            # Store the overall statistics for the modality
            overall_stats[modality] = {
                "min": overall_min,
                "max": overall_max,
                "mean": overall_mean,
                "std": overall_std,
                "q01": weighted_q01,
                "q99": weighted_q99,
            }

        return overall_stats

    @staticmethod
    def merge_metadata(
        metadatas: list[DatasetMetadata],
        dataset_sampling_weights: list[float],
        percentile_mixing_method: str,
    ) -> DatasetMetadata:
        """Merge multiple metadata into one."""
        # Convert to dicts
        metadata_dicts = [metadata.model_dump(mode="json") for metadata in metadatas]
        # Create a new metadata dict
        merged_metadata = {}

        # Check all metadata have the same embodiment tag
        assert all(
            metadata.embodiment_tag == metadatas[0].embodiment_tag for metadata in metadatas
        ), "All metadata must have the same embodiment tag"
        merged_metadata["embodiment_tag"] = metadatas[0].embodiment_tag

        # Merge the dataset statistics
        dataset_statistics = {}
        dataset_statistics["state"] = LeRobotMixtureDataset.compute_overall_statistics(
            per_task_stats=[m["statistics"]["state"] for m in metadata_dicts],
            dataset_sampling_weights=dataset_sampling_weights,
            percentile_mixing_method=percentile_mixing_method,
        )
        dataset_statistics["action"] = LeRobotMixtureDataset.compute_overall_statistics(
            per_task_stats=[m["statistics"]["action"] for m in metadata_dicts],
            dataset_sampling_weights=dataset_sampling_weights,
            percentile_mixing_method=percentile_mixing_method,
        )
        merged_metadata["statistics"] = dataset_statistics

        # Merge the modality configs
        modality_configs = defaultdict(set)
        for metadata in metadata_dicts:
            for modality, configs in metadata["modalities"].items():
                modality_configs[modality].add(json.dumps(configs))
        merged_metadata["modalities"] = {}
        for modality, configs in modality_configs.items():
            # Check that all modality configs correspond to the same tag matches
            assert (
                len(configs) == 1
            ), f"Multiple modality configs for modality {modality}: {list(configs)}"
            merged_metadata["modalities"][modality] = json.loads(configs.pop())

        return DatasetMetadata.model_validate(merged_metadata)

    def update_metadata(self, metadata_config: dict, cached_statistics_path: Path | str | None = None) -> None:
        """
        Merge multiple metadatas into one and set the transforms with the merged metadata.

        Args:
            metadata_config (dict): Configuration for the metadata.
                "percentile_mixing_method": The method to mix the percentiles, either "weighted_average" or "min_max".
                    weighted_average: Use the weighted average of the percentiles using the weight used in sampling the datasets.
                    min_max: Use the min of the 1st percentile and max of the 99th percentile.
        """
        # If cached path is provided, try to load and apply
        if cached_statistics_path is not None:
            try:
                cached_stats = self.load_merged_statistics(cached_statistics_path)
                self.apply_cached_statistics(cached_stats)
                return
            except (FileNotFoundError, KeyError, ValidationError) as e:
                raise e

        self.tag = EmbodimentTag.NEW_EMBODIMENT.value
        self.merged_metadata: dict[str, DatasetMetadata] = {}

        if self.disable_separate_embodiment_norm:
            # Merge statistics across ALL datasets globally, regardless of embodiment tag.
            # Only keys that are shared across ALL datasets are merged globally;
            # non-shared keys retain their per-dataset statistics.
            all_metadatas = [dataset.metadata for dataset in self.datasets]
            all_weights = [float(w) for w in self.dataset_sampling_weights]
            all_metadata_dicts = [m.model_dump(mode="json") for m in all_metadatas]

            # Find the intersection of stat keys across all datasets
            all_state_stats = [m["statistics"]["state"] for m in all_metadata_dicts]
            all_action_stats = [m["statistics"]["action"] for m in all_metadata_dicts]

            shared_state_keys = set(all_state_stats[0].keys())
            shared_action_keys = set(all_action_stats[0].keys())
            for s_stats, a_stats in zip(all_state_stats, all_action_stats):
                shared_state_keys &= set(s_stats.keys())
                shared_action_keys &= set(a_stats.keys())

            assert len(shared_state_keys) > 0 or len(shared_action_keys) > 0, (
                "No shared stat keys found across datasets. "
                "disable_separate_embodiment_norm requires at least some overlapping state or action keys."
            )

            # Merge only shared keys globally
            global_state_stats = {}
            if shared_state_keys:
                filtered_state = [{k: s[k] for k in shared_state_keys} for s in all_state_stats]
                global_state_stats = self.compute_overall_statistics(
                    per_task_stats=filtered_state,
                    dataset_sampling_weights=all_weights,
                    percentile_mixing_method=metadata_config["percentile_mixing_method"],
                )

            global_action_stats = {}
            if shared_action_keys:
                filtered_action = [{k: s[k] for k in shared_action_keys} for s in all_action_stats]
                global_action_stats = self.compute_overall_statistics(
                    per_task_stats=filtered_action,
                    dataset_sampling_weights=all_weights,
                    percentile_mixing_method=metadata_config["percentile_mixing_method"],
                )

            # Build per-tag metadata entries: start with each dataset's own stats,
            # then override shared keys with the globally merged stats.
            for dataset in self.datasets:
                tag = dataset.tag
                if tag not in self.merged_metadata:
                    metadata_dict = dataset.metadata.model_dump(mode="json")
                    # Keep per-dataset stats for non-shared keys, override shared keys
                    metadata_dict["statistics"]["state"].update(global_state_stats)
                    metadata_dict["statistics"]["action"].update(global_action_stats)
                    self.merged_metadata[tag] = DatasetMetadata.model_validate(metadata_dict)

            for dataset in self.datasets:
                dataset.set_transforms_metadata(self.merged_metadata[dataset.tag])

            print(f"[update_metadata] Global normalization: merged stats across all {len(self.datasets)} datasets "
                  f"({len(self.merged_metadata)} embodiment tags: {list(self.merged_metadata.keys())}). "
                  f"Shared state keys: {sorted(shared_state_keys)}, shared action keys: {sorted(shared_action_keys)}")
        else:
            # Default: group metadata by tag, merge within each embodiment group separately.
            all_metadatas_and_weights: dict[str, list[tuple[DatasetMetadata, float]]] = {}
            for dataset_idx, dataset in enumerate(self.datasets):
                if dataset.tag not in all_metadatas_and_weights:
                    all_metadatas_and_weights[dataset.tag] = []
                all_metadatas_and_weights[dataset.tag].append(
                    (dataset.metadata, float(self.dataset_sampling_weights[dataset_idx]))
                )
            for tag, metadatas_and_weights in all_metadatas_and_weights.items():
                metadatas = [mw[0] for mw in metadatas_and_weights]
                weights = [mw[1] for mw in metadatas_and_weights]
                self.merged_metadata[tag] = self.merge_metadata(
                    metadatas=metadatas,
                    dataset_sampling_weights=weights,
                    percentile_mixing_method=metadata_config["percentile_mixing_method"],
                )
            for dataset in self.datasets:
                dataset.set_transforms_metadata(self.merged_metadata[dataset.tag])

    def save_dataset_statistics(self, save_path: Path | str, format: str = "json") -> None:
        """
        Save merged dataset statistics to specified path in the required format.
        Only includes statistics for keys that are actually used in the datasets.
        Gripper-related keys will be placed at the end.
        
        Args:
            save_path (Path | str): Path to save the statistics file
            format (str): Save format, currently only supports "json"
        """
        save_path_unmerged = str(save_path).replace(".json", "_unmerged.json")
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        metadata_json = {}
        metadata_json.update(
            {
                tag: metadata.model_dump(mode="json")
                for tag, metadata in self.merged_metadata.items()
            }
                )
        # Persist normalization config so finetuning can enforce consistency
        metadata_json["metadata"] = {
            "disable_separate_embodiment_norm": self.disable_separate_embodiment_norm,
        }

        with open(save_path_unmerged , "w") as f:
                json.dump(metadata_json, f, indent=4)
        print(f"Unmerged dataset statistics saved to: {save_path_unmerged}")

        # Build the data structure to save
        statistics_data = {}
        
        # Collect actually used keys from all datasets
        all_used_action_keys = set()
        all_used_state_keys = set()
        
        for dataset in self.datasets:
            used_action_keys, used_state_keys = get_used_modality_keys(dataset.modality_keys)
            all_used_action_keys.update(used_action_keys)
            all_used_state_keys.update(used_state_keys)
        
        # Organize statistics by tag
        for tag, merged_metadata in self.merged_metadata.items():
            tag_stats = {}
            
            # Process action statistics
            if hasattr(merged_metadata.statistics, 'action') and merged_metadata.statistics.action:
                action_stats = merged_metadata.statistics.action
                
                # Filter and reorder keys
                non_gripper_keys = []
                gripper_keys = []
                
                for key in action_stats.keys():
                    if key in all_used_action_keys:
                        if "gripper" in key.lower():
                            gripper_keys.append(key)
                        else:
                            non_gripper_keys.append(key)
                
                reordered_keys = non_gripper_keys + gripper_keys
                
                filtered_action_stats = {}
                for key in reordered_keys:
                    filtered_action_stats[key] = action_stats[key]
                
                if filtered_action_stats:
                    combined_action_stats = combine_modality_stats(filtered_action_stats)
                    
                    mask = generate_action_mask_for_used_keys(
                        merged_metadata.modalities.action, filtered_action_stats.keys()
                    )
                    combined_action_stats["mask"] = mask
                    
                    tag_stats["action"] = combined_action_stats
            
            # Process state statistics
            if hasattr(merged_metadata.statistics, 'state') and merged_metadata.statistics.state:
                state_stats = merged_metadata.statistics.state
                
                # Filter and reorder keys
                non_gripper_keys = []
                gripper_keys = []
                
                for key in state_stats.keys():
                    if key in all_used_state_keys:
                        if "gripper" in key.lower():
                            gripper_keys.append(key)
                        else:
                            non_gripper_keys.append(key)
                
                reordered_keys = non_gripper_keys + gripper_keys
                
                filtered_state_stats = {}
                for key in reordered_keys:
                    filtered_state_stats[key] = state_stats[key]
                
                if filtered_state_stats:
                    combined_state_stats = combine_modality_stats(filtered_state_stats)
                    tag_stats["state"] = combined_state_stats
            
            # Add dataset counts
            tag_stats.update(self._get_dataset_counts(tag))
            
            statistics_data[tag] = tag_stats
        
        # Save file
        if format.lower() == "json":
            if not str(save_path).endswith('.json'):
                save_path = save_path.with_suffix('.json')
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(statistics_data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}. Currently only 'json' is supported.")
        
        print(f"Merged dataset statistics saved to: {save_path}")
        print(f"Used action keys (reordered): {list(all_used_action_keys)}")
        print(f"Used state keys (reordered): {list(all_used_state_keys)}")

    def _combine_modality_stats(self, modality_stats: dict) -> dict:
        """Backward compatibility wrapper."""
        return combine_modality_stats(modality_stats)

    def _generate_action_mask_for_used_keys(self, action_modalities: dict, used_action_keys_ordered) -> list[bool]:
        """Backward compatibility wrapper."""
        return generate_action_mask_for_used_keys(action_modalities, used_action_keys_ordered)

    def _get_dataset_counts(self, tag: str) -> dict:
        """
        Get dataset count information for specified tag.
        
        Args:
            tag (str): embodiment tag
            
        Returns:
            dict: Dictionary containing num_transitions and num_trajectories
        """
        num_transitions = 0
        num_trajectories = 0
        
        # Count dataset information belonging to this tag
        for dataset in self.datasets:
            if dataset.tag == tag:
                num_transitions += len(dataset)
                num_trajectories += len(dataset.trajectory_ids)
        
        return {
            "num_transitions": num_transitions,
            "num_trajectories": num_trajectories
        }

    @classmethod
    def load_merged_statistics(cls, load_path: Path | str) -> dict:
        """
        Load merged dataset statistics from file.
        
        Args:
            load_path (Path | str): Path to the statistics file
            
        Returns:
            dict: Dictionary containing merged statistics
        """
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"Statistics file not found: {load_path}")
        
        if load_path.suffix.lower() == '.json':
            with open(load_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif load_path.suffix.lower() == '.pkl':
            import pickle
            with open(load_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {load_path.suffix}")

    def apply_cached_statistics(self, cached_statistics: dict) -> None:
        """
        Apply cached statistics to avoid recomputation.
        
        This method expects the unmerged format (dataset_statistics_unmerged.json),
        which contains full DatasetMetadata for each embodiment tag.
        
        Args:
            cached_statistics (dict): Statistics loaded from file (unmerged format)
        """
        # Enforce consistency of disable_separate_embodiment_norm between pretrain and finetune
        cached_meta = cached_statistics.get("metadata", {})
        cached_flag = cached_meta.get("disable_separate_embodiment_norm", False)
        if cached_flag != self.disable_separate_embodiment_norm:
            raise RuntimeError(
                f"Cached statistics were saved with disable_separate_embodiment_norm={cached_flag}, "
                f"but current run uses disable_separate_embodiment_norm={self.disable_separate_embodiment_norm}. "
                f"These must match to ensure consistent normalization between pretrain and finetune."
            )

        # Apply cached statistics - expect unmerged format with full DatasetMetadata per tag
        self.merged_metadata = {}
        for tag, metadata_data in cached_statistics.items():
            if tag == "metadata":  # Skip metadata field if present
                continue
            
            # The unmerged format has full DatasetMetadata structure
            # Check if this looks like the unmerged format (has 'statistics' and 'modalities' keys)
            if "statistics" in metadata_data and "modalities" in metadata_data:
                # Unmerged format - directly validate as DatasetMetadata
                self.merged_metadata[tag] = DatasetMetadata.model_validate(metadata_data)
            else:
                # This might be the old combined format - skip with warning
                raise ValueError(f"Tag '{tag}' does not have expected unmerged format.")
        
        if not self.merged_metadata:
            raise ValueError(
                "No valid metadata found in cached statistics. "
                "Please use dataset_statistics_unmerged.json which contains full metadata format."
            )
        
        # Collect tags that are missing from the cache and compute fresh metadata for them
        uncached_tags: dict[str, list[tuple[DatasetMetadata, float]]] = {}
        for dataset_idx, dataset in enumerate(self.datasets):
            if dataset.tag not in self.merged_metadata:
                if dataset.tag not in uncached_tags:
                    uncached_tags[dataset.tag] = []
                uncached_tags[dataset.tag].append(
                    (dataset.metadata, float(self.dataset_sampling_weights[dataset_idx]))
                )

        for tag, metadatas_and_weights in uncached_tags.items():
            metadatas = [mw[0] for mw in metadatas_and_weights]
            weights = [mw[1] for mw in metadatas_and_weights]
            self.merged_metadata[tag] = self.merge_metadata(
                metadatas=metadatas,
                dataset_sampling_weights=weights,
                percentile_mixing_method="weighted_average",
            )
            print(f"[apply_cached_statistics] New embodiment tag '{tag}' not in cache — "
                  f"computed fresh statistics from {len(metadatas)} datasets")

        # Update transforms metadata for each dataset
        for dataset in self.datasets:
            dataset.set_transforms_metadata(self.merged_metadata[dataset.tag])
        
        print(f"Applied cached statistics for {len(self.merged_metadata)} embodiment tags "
              f"({len(uncached_tags)} computed fresh).")
