"""
Relative Action Transform for converting absolute actions to relative representations.

This module provides functionality to transform absolute end-effector (EEF) actions
into relative representations for robot learning tasks.

Supported Relative Types:
    - "relative": All actions are expressed relative to the first observation (S0)
                  in the action chunk. Good for understanding overall trajectory.
    - "delta": Each action is expressed relative to the previous pose.
               Better for capturing local motion patterns.

Supported Frame Types:
    - "local": Delta is expressed in the reference pose's local coordinate frame.
               Rotation-invariant - good for learning manipulation skills that
               should work regardless of robot orientation.
    - "global": Delta is expressed in the world/base coordinate frame.
               Useful when global directions matter (e.g., "move up").

Example Transform Pipeline:
    StateActionToTensor → RelativeActionTransform → StateActionTransform → ConcatTransform

Key Components:
    - StreamingStats: Memory-efficient statistics accumulator using Welford's algorithm
    - calculate_relative_dataset_statistics: Parallel computation of dataset statistics
    - RelativeActionTransform: The main transform class (invertible)
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
import pytorch3d.transforms as pt
import torch
from pydantic import Field, PrivateAttr
from tqdm import tqdm

from .base import InvertibleModalityTransform


# ============================================================================
# Streaming Statistics for Memory-Efficient Computation
# ============================================================================


@dataclass
class StreamingStats:
    """Streaming statistics accumulator using Welford's algorithm.
    
    This class enables memory-efficient computation of statistics over large
    datasets without needing to load all data into memory at once. It supports:
    
    - Online mean and variance computation using Welford's algorithm
    - Streaming min/max tracking
    - Reservoir sampling for quantile estimation
    
    The statistics can be computed in parallel by multiple workers and then
    merged together using the merge() method (via Chan's parallel algorithm).
    
    Attributes:
        dim: Dimensionality of the data vectors
        reservoir_size: Maximum size of reservoir for quantile sampling
        count: Total number of samples seen
        mean: Running mean of all samples
        M2: Sum of squared differences from mean (for variance)
        min_val: Element-wise minimum values seen
        max_val: Element-wise maximum values seen
        reservoir: List of sampled data points for quantile estimation
    
    Example:
        >>> stats = StreamingStats(dim=6, reservoir_size=10000)
        >>> for batch in data_loader:
        ...     stats.update_batch(batch)
        >>> print(f"Mean: {stats.mean}, Std: {stats.std}")
    """
    
    dim: int
    reservoir_size: int = 50000
    count: int = 0
    mean: np.ndarray = None
    M2: np.ndarray = None
    min_val: np.ndarray = None
    max_val: np.ndarray = None
    reservoir: list = field(default_factory=list)
    _reservoir_count: int = 0
    
    def __post_init__(self):
        """Initialize arrays after dataclass creation."""
        self.mean = np.zeros(self.dim, dtype=np.float64)
        self.M2 = np.zeros(self.dim, dtype=np.float64)
        self.min_val = np.full(self.dim, np.inf, dtype=np.float64)
        self.max_val = np.full(self.dim, -np.inf, dtype=np.float64)
        self.reservoir = []
        self._reservoir_count = 0
    
    def update_batch(self, x: np.ndarray):
        """Update statistics with a batch of samples.
        
        Uses Chan's parallel algorithm for efficient batch updates of mean
        and variance. More efficient than updating one sample at a time.
        
        Args:
            x: Array of shape [N, dim] containing N samples. If 1D, will be
               reshaped to [1, dim].
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        n = x.shape[0]
        if n == 0:
            return
        
        # Compute batch statistics
        batch_count = n
        batch_mean = np.mean(x, axis=0)
        batch_M2 = np.var(x, axis=0, ddof=0) * n
        
        # Merge with existing statistics using Chan's parallel algorithm
        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean.astype(np.float64)
            self.M2 = batch_M2.astype(np.float64)
        else:
            combined_count = self.count + batch_count
            delta = batch_mean - self.mean
            self.mean = (self.count * self.mean + batch_count * batch_mean) / combined_count
            self.M2 = self.M2 + batch_M2 + delta**2 * self.count * batch_count / combined_count
            self.count = combined_count
        
        # Update min/max
        self.min_val = np.minimum(self.min_val, np.min(x, axis=0))
        self.max_val = np.maximum(self.max_val, np.max(x, axis=0))
        
        # Reservoir sampling for quantile estimation
        for row in x:
            self._reservoir_count += 1
            if len(self.reservoir) < self.reservoir_size:
                self.reservoir.append(row.copy())
            else:
                # Random replacement with probability reservoir_size / _reservoir_count
                j = np.random.randint(0, self._reservoir_count)
                if j < self.reservoir_size:
                    self.reservoir[j] = row.copy()
    
    def merge(self, other: 'StreamingStats') -> 'StreamingStats':
        """Merge another StreamingStats into this one.
        
        Uses Chan's parallel algorithm for combining mean and variance.
        This allows parallel computation where each worker computes partial
        statistics that are then merged together.
        
        Args:
            other: Another StreamingStats instance to merge
            
        Returns:
            self (for method chaining)
        """
        if other.count == 0:
            return self
        
        if self.count == 0:
            # Copy all values from other
            self.count = other.count
            self.mean = other.mean.copy()
            self.M2 = other.M2.copy()
            self.min_val = other.min_val.copy()
            self.max_val = other.max_val.copy()
            self.reservoir = other.reservoir.copy()
            self._reservoir_count = other._reservoir_count
            return self
        
        # Merge mean and M2 using Chan's parallel variance algorithm
        combined_count = self.count + other.count
        delta = other.mean - self.mean
        
        self.mean = (self.count * self.mean + other.count * other.mean) / combined_count
        self.M2 = self.M2 + other.M2 + delta**2 * self.count * other.count / combined_count
        self.count = combined_count
        
        # Merge min/max
        self.min_val = np.minimum(self.min_val, other.min_val)
        self.max_val = np.maximum(self.max_val, other.max_val)
        
        # Merge reservoirs with random selection to maintain proper sampling
        combined_reservoir = self.reservoir + other.reservoir
        combined_res_count = self._reservoir_count + other._reservoir_count
        
        if len(combined_reservoir) > self.reservoir_size:
            indices = np.random.choice(len(combined_reservoir), self.reservoir_size, replace=False)
            self.reservoir = [combined_reservoir[i] for i in indices]
        else:
            self.reservoir = combined_reservoir
        
        self._reservoir_count = combined_res_count
        
        return self
    
    @property
    def std(self) -> np.ndarray:
        """Compute standard deviation from accumulated M2."""
        if self.count < 2:
            return np.zeros(self.dim, dtype=np.float64)
        return np.sqrt(self.M2 / self.count)
    
    def quantile(self, q: float) -> np.ndarray:
        """Estimate quantile from reservoir sample.
        
        Args:
            q: Quantile value between 0 and 1 (e.g., 0.5 for median)
            
        Returns:
            Estimated quantile values for each dimension
        """
        if len(self.reservoir) == 0:
            return np.zeros(self.dim, dtype=np.float64)
        reservoir_array = np.array(self.reservoir, dtype=np.float64)
        return np.quantile(reservoir_array, q, axis=0)
    
    def to_stats_dict(self) -> dict:
        """Convert to standard statistics dictionary format.
        
        Returns:
            Dictionary with keys: mean, std, min, max, q01, q99
        """
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "min": self.min_val.tolist(),
            "max": self.max_val.tolist(),
            "q01": self.quantile(0.01).tolist(),
            "q99": self.quantile(0.99).tolist(),
        }


# ============================================================================
# Helper Functions for Modality Metadata
# ============================================================================


_POSITION_SUFFIXES = ("_position", "_pos")
_ROTATION_SUFFIXES = ("_rotation", "_rotation_6d", "_orientation_6d")


def _is_position_key(key: str) -> bool:
    """Check if *key* represents position data."""
    return any(key.endswith(suffix) for suffix in _POSITION_SUFFIXES)


def _is_eef_key(key: str) -> bool:
    """Check if a key represents end-effector (EEF) data that should be transformed.
    
    EEF keys include position and rotation/orientation components that need
    relative transformation.  Non-EEF keys (like gripper state, head_6d,
    waist_6d, joint, effector) are passed through unchanged.
    
    Recognised suffixes:
        Position: ``_position``, ``_pos``
        Rotation: ``_rotation``, ``_rotation_6d``, ``_orientation_6d``
    
    Args:
        key: Sub-key name from modality metadata
        
    Returns:
        True if this key should be transformed to relative representation
    """
    if key.endswith("_position"):
        return True
    if key.endswith("_pos"):
        return "_eef_" in key
    if key.endswith("_orientation_6d"):
        return "_eef_" in key
    return key.endswith("_rotation") or key.endswith("_rotation_6d")


def _is_rotation_key(key: str) -> bool:
    """Check if *key* is a rotation/orientation EEF key."""
    return key.endswith("_6d") or key.endswith("_rotation")


def _get_subkey_indices(modality_meta: Any, modality: str) -> dict[str, tuple[int, int, str]]:
    """Extract sub-key indices from modality metadata.
    
    Parses the modality metadata to get the start/end indices for each sub-key
    within the combined action or state vector.
    
    Args:
        modality_meta: LeRobotModalityMetadata object
        modality: Either "action" or "state"
        
    Returns:
        Dictionary mapping sub-key name to (start_idx, end_idx, original_column_name)
    """
    modality_dict = getattr(modality_meta, modality)
    indices = {}
    for subkey, meta in modality_dict.items():
        original_key = meta.original_key if meta.original_key else modality
        indices[subkey] = (meta.start, meta.end, original_key)
    return indices


def _get_rotation_key_for_position_stats(position_key: str, action_indices: dict) -> str:
    """Get the corresponding rotation key for a position key.
    
    For local frame transforms, positions need to be transformed using the
    corresponding rotation. This function finds the matching rotation key.
    
    Tries suffixes in order of preference: 6D representations first (needed
    by _rot6d_to_matrix), then generic rotation/orientation fallbacks.
    
    Args:
        position_key: Position sub-key (e.g., "leftHand_position" or "left_eef_position")
        action_indices: Dictionary of all action sub-key indices
        
    Returns:
        The matching rotation key
        
    Raises:
        ValueError: If no matching rotation key is found
    """
    if position_key.endswith("_position"):
        prefix = position_key.removesuffix("_position")
    elif position_key.endswith("_pos"):
        prefix = position_key.removesuffix("_pos")
    else:
        raise ValueError(f"Position key must end with '_position' or '_pos', got '{position_key}'")
    candidates = [
        f"{prefix}_rotation_6d",
        f"{prefix}_orientation_6d",
        f"{prefix}_6d",
        f"{prefix}_rotation",
    ]
    for rot_key in candidates:
        if rot_key in action_indices:
            return rot_key
    raise ValueError(f"No rotation key found for position key '{position_key}'. "
                     f"Tried: {candidates}. Available keys: {sorted(action_indices.keys())}")


def _get_original_column_name(indices: dict[str, tuple[int, int, str]]) -> str:
    """Get the original column name from indices dict.
    
    All sub-keys should have the same original column name (e.g., "action" or "state").
    
    Args:
        indices: Dictionary from _get_subkey_indices
        
    Returns:
        The common original column name
        
    Raises:
        AssertionError: If column names are inconsistent
    """
    col_names = set(v[2] for v in indices.values())
    assert len(col_names) == 1, f"Inconsistent original column names: {col_names}"
    return col_names.pop()


def _remap_indices(indices: dict[str, tuple[int, int, str]]) -> dict[str, tuple[int, int, str]]:
    """Remap sub-key indices to contiguous positions in a combined vector.

    For single-column data the result is equivalent to the original (sub-keys
    already address the same vector). For multi-column data each sub-key's
    slice is laid out sequentially in sorted key order so that a combined
    vector can be built from heterogeneous columns.

    Args:
        indices: Dict of sub-key -> (start, end, original_key)

    Returns:
        Dict of sub-key -> (new_start, new_end, original_key) with
        contiguous, non-overlapping ranges covering [0, total_dim).
    """
    remapped = {}
    offset = 0
    for key in sorted(indices.keys()):
        start, end, orig_col = indices[key]
        dim = end - start
        remapped[key] = (offset, offset + dim, orig_col)
        offset += dim
    return remapped


def _build_combined_array_for_episode(
    df: "pd.DataFrame",
    indices: dict[str, tuple[int, int, str]],
) -> tuple[np.ndarray, dict[str, tuple[int, int, str]]]:
    """Assemble a combined vector from potentially multiple DataFrame columns.

    Each sub-key in *indices* specifies (start, end, original_key) where
    *original_key* is the DataFrame column and start:end is the slice within
    that column. This function reads each slice, concatenates them in sorted
    key order, and returns the combined array together with a remapped index
    dict that addresses the combined vector.

    Works for both single-column and multi-column data.

    Args:
        df: DataFrame with at least the columns referenced by indices.
        indices: Dict of sub-key -> (start, end, original_key).

    Returns:
        (combined_array, remapped_indices) where combined_array has shape
        [n_rows, total_dim] and remapped_indices maps each sub-key to its
        position in the combined vector.
    """
    n_rows = len(df)
    remapped = _remap_indices(indices)

    total_dim = sum(end - start for start, end, _ in indices.values())
    combined = np.empty((n_rows, total_dim), dtype=np.float32)

    # Cache column arrays to avoid repeated stacking
    _col_cache: dict[str, np.ndarray] = {}

    for key in sorted(indices.keys()):
        orig_start, orig_end, orig_col = indices[key]
        if orig_col not in _col_cache:
            stacked = np.stack(
                [np.asarray(x, dtype=np.float32) for x in df[orig_col].values]
            )
            if stacked.ndim == 1:
                stacked = stacked[:, np.newaxis]
            _col_cache[orig_col] = stacked
        new_start, new_end, _ = remapped[key]
        combined[:, new_start:new_end] = _col_cache[orig_col][:, orig_start:orig_end]

    return combined, remapped


def _split_stats_to_columns(
    combined_stats: dict[str, list],
    remapped_indices: dict[str, tuple[int, int, str]],
    original_indices: dict[str, tuple[int, int, str]],
) -> dict[str, dict[str, list]]:
    """Split combined-vector statistics back into per-original_key column format.

    This reverses the assembly done by _build_combined_array_for_episode: it
    reads each sub-key's slice from the combined stats (using remapped_indices)
    and writes it into the correct position in the per-column stats dict (using
    original_indices). The result is keyed by original_key and can be consumed
    by _get_metadata which does ``stats[original_key][stat_name][..., start:end]``.

    Args:
        combined_stats: Stats dict with stat_name -> list (2D or 1D).
        remapped_indices: Dict of sub-key -> (start, end, original_key) in combined vector.
        original_indices: Dict of sub-key -> (start, end, original_key) from modality metadata.

    Returns:
        Dict of original_key -> {stat_name: array-as-list} with correct dimensions.
    """
    # Determine max dimension for each original column
    col_dims: dict[str, int] = {}
    for _key, (start, end, col) in original_indices.items():
        col_dims[col] = max(col_dims.get(col, 0), end)

    result: dict[str, dict[str, np.ndarray]] = {}

    for stat_name, stat_values in combined_stats.items():
        stat_array = np.array(stat_values)
        is_2d = stat_array.ndim == 2

        for col, dim in col_dims.items():
            if col not in result:
                result[col] = {}
            if stat_name not in result[col]:
                if is_2d:
                    result[col][stat_name] = np.zeros((stat_array.shape[0], dim))
                else:
                    result[col][stat_name] = np.zeros(dim)

        for key in original_indices:
            orig_start, orig_end, orig_col = original_indices[key]
            new_start, new_end, _ = remapped_indices[key]
            if is_2d:
                result[orig_col][stat_name][:, orig_start:orig_end] = stat_array[:, new_start:new_end]
            else:
                result[orig_col][stat_name][orig_start:orig_end] = stat_array[new_start:new_end]

    # Convert numpy arrays to lists for JSON serialization
    return {
        col: {stat_name: arr.tolist() for stat_name, arr in stats.items()}
        for col, stats in result.items()
    }


# ============================================================================
# Rotation and Position Transform Functions (NumPy, for Statistics)
# ============================================================================


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation representation to 3x3 rotation matrix.
    
    Uses the continuous 6D rotation representation from:
    "On the Continuity of Rotation Representations in Neural Networks"
    
    Args:
        rot6d: 6D rotation representation, shape [6]
        
    Returns:
        Rotation matrix, shape [3, 3]
    """
    if rot6d.shape[-1] != 6:
        raise ValueError(f"Expected 6D rotation, got shape {rot6d.shape}")
    tensor = torch.from_numpy(rot6d.reshape(1, 6).astype(np.float32))
    matrix = pt.rotation_6d_to_matrix(tensor)
    return matrix[0].numpy()


def _matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to 6D rotation representation.
    
    Args:
        matrix: Rotation matrix, shape [3, 3]
        
    Returns:
        6D rotation representation, shape [6]
    """
    tensor = torch.from_numpy(matrix.reshape(1, 3, 3).astype(np.float32))
    rot6d = pt.matrix_to_rotation_6d(tensor)
    return rot6d[0].numpy()


def _compute_relative_rotation(
    R_curr: np.ndarray,
    R_ref: np.ndarray,
    frame_type: str,
) -> np.ndarray:
    """Compute relative rotation between two rotation matrices.
    
    Args:
        R_curr: Current (target) rotation matrix [3, 3]
        R_ref: Reference rotation matrix [3, 3]
        frame_type: "local" or "global"
            - local: delta_R = R_ref.T @ R_curr (rotation in reference frame)
            - global: delta_R = R_curr @ R_ref.T (rotation in world frame)
        
    Returns:
        Relative rotation in 6D representation [6]
    """
    if frame_type == "local":
        delta_R = R_ref.T @ R_curr
    else:
        delta_R = R_curr @ R_ref.T
    
    return _matrix_to_rot6d(delta_R)


def _compute_relative_position(
    action_pos: np.ndarray,
    ref_pos: np.ndarray,
    frame_type: str,
    ref_rotation: np.ndarray,
) -> np.ndarray:
    """Compute relative position.
    
    Args:
        action_pos: Current position [3]
        ref_pos: Reference position [3]
        frame_type: "local" or "global"
            - local: delta expressed in reference frame's coordinates
            - global: delta expressed in world coordinates
        ref_rotation: Reference rotation matrix [3, 3] (used for local frame)
        
    Returns:
        Relative position [3]
    """
    delta = action_pos - ref_pos
    
    if frame_type == "local":
        delta = ref_rotation.T @ delta
    
    return delta


# ============================================================================
# Chunk Transformation Functions (for Statistics Computation)
# ============================================================================


def _transform_rotation_sequence(
    action_rot: np.ndarray,
    ref_state_rot: np.ndarray,
    relative_type: str,
    frame_type: str,
) -> np.ndarray:
    """Transform a sequence of rotations to relative representation.
    
    Args:
        action_rot: Action rotations [T, 6] in 6D representation
        ref_state_rot: Reference state rotation [6] (S0)
        relative_type: "relative" (all to S0) or "delta" (to previous)
        frame_type: "local" or "global"
        
    Returns:
        Relative rotations [T, 6]
    """
    T = action_rot.shape[0]
    result = np.zeros_like(action_rot)
    
    ref_matrix = _rot6d_to_matrix(ref_state_rot)
    
    if relative_type == "relative":
        # All rotations relative to S0
        for t in range(T):
            action_matrix = _rot6d_to_matrix(action_rot[t])
            result[t] = _compute_relative_rotation(action_matrix, ref_matrix, frame_type)
    else:
        # Delta: each rotation relative to previous
        prev_matrix = ref_matrix
        for t in range(T):
            action_matrix = _rot6d_to_matrix(action_rot[t])
            result[t] = _compute_relative_rotation(action_matrix, prev_matrix, frame_type)
            prev_matrix = action_matrix
    
    return result


def _transform_position_sequence(
    action_pos: np.ndarray,
    ref_state_pos: np.ndarray,
    ref_rotation_matrix: np.ndarray,
    action_rot_matrices: np.ndarray,
    relative_type: str,
    frame_type: str,
) -> np.ndarray:
    """Transform a sequence of positions to relative representation.
    
    Args:
        action_pos: Action positions [T, 3]
        ref_state_pos: Reference state position [3] (S0)
        ref_rotation_matrix: Reference rotation matrix [3, 3] (S0 rotation, for local frame)
        action_rot_matrices: Absolute rotation matrices at each timestep [T, 3, 3]
                            (needed for delta+local mode to update reference frame)
        relative_type: "relative" (all to S0) or "delta" (to previous)
        frame_type: "local" or "global"
        
    Returns:
        Relative positions [T, 3]
    """
    T = action_pos.shape[0]
    result = np.zeros_like(action_pos)
    
    if relative_type == "relative":
        # All positions relative to S0
        for t in range(T):
            result[t] = _compute_relative_position(
                action_pos[t], ref_state_pos, frame_type, ref_rotation_matrix
            )
    else:
        # Delta: each position relative to previous
        prev_pos = ref_state_pos
        prev_rot = ref_rotation_matrix
        
        for t in range(T):
            result[t] = _compute_relative_position(action_pos[t], prev_pos, frame_type, prev_rot)
            prev_pos = action_pos[t]
            # For delta+local mode, update the reference rotation to current pose
            if frame_type == "local":
                prev_rot = action_rot_matrices[t]
    
    return result


def _transform_chunk_actions(
    action_array: np.ndarray,
    state_array: np.ndarray,
    action_indices: dict[str, tuple[int, int, str]],
    state_indices: dict[str, tuple[int, int, str]],
    eef_keys: set[str],
    relative_type: str,
    frame_type: str,
) -> np.ndarray:
    """Transform an action chunk to relative representation.
    
    The reference frame (S0) is the first state in the chunk (state_array[0]).
    This matches training behavior where each action chunk uses its first
    observation as the reference for relative actions.
    
    Processing order:
    1. First transform all rotations (needed for local frame position transforms)
    2. Then transform all positions using the corresponding rotation data
    
    Args:
        action_array: Combined action array [chunk_len, D_action]
        state_array: Combined state array [chunk_len, D_state]
        action_indices: Dict of sub-key -> (start, end, col_name) for actions
        state_indices: Dict of sub-key -> (start, end, col_name) for states
        eef_keys: Set of EEF keys to transform (position and rotation keys)
        relative_type: "relative" or "delta"
        frame_type: "local" or "global"
        
    Returns:
        Transformed action array [chunk_len, D_action] with EEF keys in
        relative representation, non-EEF keys unchanged
    """
    T = action_array.shape[0]
    result = action_array.copy()
    
    # First pass: transform rotations (needed for position transforms in local frame)
    for subkey in eef_keys:
        if not _is_rotation_key(subkey):
            continue
        
        action_start, action_end, _ = action_indices[subkey]
        state_start, state_end, _ = state_indices[subkey]
        
        action_rot = action_array[:, action_start:action_end]
        state_rot = state_array[:, state_start:state_end]
        ref_state_rot = state_rot[0]  # S0 rotation
        if action_rot.shape[-1] != 6 or state_rot.shape[-1] != 6:
            raise ValueError(
                f"Rotation key '{subkey}' must be 6D for relative stats, "
                f"got action dim {action_rot.shape[-1]} and state dim {state_rot.shape[-1]}"
            )
        
        relative_rot = _transform_rotation_sequence(
            action_rot, ref_state_rot, relative_type, frame_type
        )
        result[:, action_start:action_end] = relative_rot
    
    # Second pass: transform positions
    for subkey in eef_keys:
        if not _is_position_key(subkey):
            continue
        
        action_start, action_end, _ = action_indices[subkey]
        state_start, state_end, _ = state_indices[subkey]
        
        action_pos = action_array[:, action_start:action_end]
        state_pos = state_array[:, state_start:state_end]
        ref_state_pos = state_pos[0]  # S0 position
        
        # Get corresponding rotation key for this position
        rot_key = _get_rotation_key_for_position_stats(subkey, action_indices)
        rot_state_start, rot_state_end, _ = state_indices[rot_key]
        rot_action_start, rot_action_end, _ = action_indices[rot_key]
        rot_state_dim = rot_state_end - rot_state_start
        rot_action_dim = rot_action_end - rot_action_start
        if rot_state_dim != 6 or rot_action_dim != 6:
            raise ValueError(
                f"Position key '{subkey}' matched rotation key '{rot_key}', "
                f"but relative stats require a 6D rotation key. "
                f"Got action dim {rot_action_dim} and state dim {rot_state_dim}."
            )
        
        # Reference rotation (S0)
        ref_state_rot6d = state_array[0, rot_state_start:rot_state_end]
        ref_rotation_matrix = _rot6d_to_matrix(ref_state_rot6d)
        
        # Absolute rotation matrices at each timestep (needed for delta mode)
        action_rot_matrices = np.stack([
            _rot6d_to_matrix(action_array[t, rot_action_start:rot_action_end])
            for t in range(T)
        ])
        
        relative_pos = _transform_position_sequence(
            action_pos, ref_state_pos, ref_rotation_matrix,
            action_rot_matrices, relative_type, frame_type
        )
        result[:, action_start:action_end] = relative_pos
    
    return result


def _extract_eef_components(
    action_row: np.ndarray,
    action_indices: dict[str, tuple[int, int, str]],
    eef_keys: set[str],
) -> np.ndarray:
    """Extract EEF components from an action row in consistent order.
    
    Components are extracted in sorted key order to ensure consistent
    concatenation across different calls.
    
    Args:
        action_row: Full action row [D_action]
        action_indices: Dict of sub-key -> (start, end, col_name)
        eef_keys: Set of EEF keys to extract
        
    Returns:
        Concatenated EEF components in sorted key order
    """
    components = []
    for key in sorted(eef_keys):
        start, end, _ = action_indices[key]
        components.append(action_row[start:end])
    return np.concatenate(components)


# ============================================================================
# Multiprocessing Worker Function
# ============================================================================


def _process_episode_batch(args) -> dict:
    """Process a batch of episodes and return streaming statistics.
    
    This function is designed to be called by multiprocessing workers.
    Each worker processes a subset of episodes and returns serialized
    streaming statistics that can be merged with other workers' results.
    
    Statistics are computed PER POSITION in the action chunk.
    
    Args:
        args: Tuple of (episode_data_list, config_dict)
            - episode_data_list: List of (episode_idx, action_array, state_array) tuples
            - config_dict: Configuration parameters including:
                - action_indices, state_indices: Sub-key index mappings
                - eef_keys, non_eef_keys: Keys to transform / pass through
                - relative_type, frame_type: Transform configuration
                - action_chunk_size: Size of action chunks
                - worker_seed: Random seed for this worker
                - reservoir_size: Size of reservoir for quantile sampling
            
    Returns:
        Dictionary containing:
            - eef_stats_per_position: List of serialized StreamingStats, one per chunk position
            - non_eef_stats: Dict of serialized StreamingStats for each non-EEF key
            - total_chunks: Number of chunks processed
    """
    episode_data_list, config = args
    
    # Unpack configuration
    action_indices = config['action_indices']
    state_indices = config['state_indices']
    eef_keys = config['eef_keys']
    non_eef_keys = config['non_eef_keys']
    relative_type = config['relative_type']
    frame_type = config['frame_type']
    action_chunk_size = config['action_chunk_size']
    worker_seed = config['worker_seed']
    reservoir_size = config.get('reservoir_size', 50000)
    
    # Set worker-specific random seed for reproducibility
    np.random.seed(worker_seed)
    
    # Calculate EEF dimension (sum of all EEF key dimensions)
    eef_dim = sum(action_indices[k][1] - action_indices[k][0] for k in sorted(eef_keys))
    
    # Initialize streaming stats for EEF data - ONE PER POSITION in the chunk
    # This is critical: position 0 has smaller deltas than position 49
    eef_stats_per_position = [
        StreamingStats(dim=eef_dim, reservoir_size=reservoir_size // action_chunk_size)
        for _ in range(action_chunk_size)
    ]
    
    # Initialize streaming stats for non-EEF data (one per key, NOT per position)
    # Non-EEF keys (like gripper) don't need per-position stats
    non_eef_streaming = {}
    for key in non_eef_keys:
        start, end, _ = action_indices[key]
        non_eef_streaming[key] = StreamingStats(dim=end - start, reservoir_size=reservoir_size)
    
    total_chunks = 0
    
    for episode_idx, action_array, state_array in tqdm(episode_data_list, desc="Processing episodes"):
        ep_len = action_array.shape[0]
        if ep_len < 2:
            continue
        
        # Update non-EEF stats directly from raw data (no transformation needed)
        for key in non_eef_keys:
            start, end, _ = action_indices[key]
            key_data = action_array[:, start:end]
            non_eef_streaming[key].update_batch(key_data)
        
        # Sample chunk starting positions to mimic training behavior
        num_valid_starts = max(1, ep_len - 1)
        num_samples = max(1, (2 * ep_len) // action_chunk_size)
        sampled_starts = np.random.randint(0, num_valid_starts, size=num_samples)
        
        for base_idx in sampled_starts:
            chunk_end = min(base_idx + action_chunk_size, ep_len)
            chunk_len = chunk_end - base_idx
            
            if chunk_len < 2:
                continue
            
            # Extract chunk data
            chunk_action = action_array[base_idx:chunk_end]
            chunk_state = state_array[base_idx:chunk_end]
            
            # Transform chunk to relative representation
            transformed_chunk = _transform_chunk_actions(
                action_array=chunk_action,
                state_array=chunk_state,
                action_indices=action_indices,
                state_indices=state_indices,
                eef_keys=eef_keys,
                relative_type=relative_type,
                frame_type=frame_type,
            )
            
            # Extract and accumulate EEF statistics PER POSITION
            if eef_keys:
                for t in range(chunk_len):
                    eef_sample = _extract_eef_components(transformed_chunk[t], action_indices, eef_keys)
                    eef_stats_per_position[t].update_batch(eef_sample)
            
            total_chunks += 1
    
    # Serialize streaming stats for return (numpy arrays are pickle-able)
    result = {
        'eef_stats_per_position': [
            {
                'dim': stats.dim,
                'count': stats.count,
                'mean': stats.mean,
                'M2': stats.M2,
                'min_val': stats.min_val,
                'max_val': stats.max_val,
                'reservoir': stats.reservoir,
                '_reservoir_count': stats._reservoir_count,
            }
            for stats in eef_stats_per_position
        ],
        'non_eef_stats': {},
        'total_chunks': total_chunks,
    }
    
    for key, stats in non_eef_streaming.items():
        result['non_eef_stats'][key] = {
            'dim': stats.dim,
            'count': stats.count,
            'mean': stats.mean,
            'M2': stats.M2,
            'min_val': stats.min_val,
            'max_val': stats.max_val,
            'reservoir': stats.reservoir,
            '_reservoir_count': stats._reservoir_count,
        }
    
    return result


def _deserialize_streaming_stats(data: dict) -> StreamingStats:
    """Deserialize streaming stats from dictionary returned by worker.
    
    Args:
        data: Dictionary with serialized StreamingStats fields
        
    Returns:
        Reconstructed StreamingStats instance
    """
    stats = StreamingStats(dim=data['dim'])
    stats.count = data['count']
    stats.mean = data['mean']
    stats.M2 = data['M2']
    stats.min_val = data['min_val']
    stats.max_val = data['max_val']
    stats.reservoir = data['reservoir']
    stats._reservoir_count = data['_reservoir_count']
    return stats


# ============================================================================
# Statistics Merging Functions
# ============================================================================


def _merge_action_statistics_per_position(
    action_dim: int,
    action_chunk_size: int,
    action_indices: dict[str, tuple[int, int, str]],
    eef_keys: set[str],
    non_eef_keys: set[str],
    eef_stats_per_position: list[dict[str, dict]],
    non_eef_stats: dict[str, dict],
) -> dict:
    """Merge EEF and non-EEF statistics into full action statistics with per-position stats.
    
    Combines the separately computed EEF (transformed, per-position) and non-EEF (raw)
    statistics into a single statistics dictionary matching the full action vector layout.
    
    IMPORTANT: EEF statistics are PER POSITION in the chunk. Each position in the
    action chunk has different statistics because relative deltas grow with distance
    from the reference.
    
    Args:
        action_dim: Total dimension of action vector
        action_chunk_size: Number of positions in the action chunk
        action_indices: Dict of sub-key -> (start, end, col_name)
        eef_keys: Set of EEF keys (transformed data)
        non_eef_keys: Set of non-EEF keys (raw data)
        eef_stats_per_position: List of per-position statistics dicts for EEF keys
            Each element is a dict[key, {"mean": [...], "std": [...], ...}]
        non_eef_stats: Statistics dict for non-EEF keys (NOT per-position)
        
    Returns:
        Merged statistics dict with mean, std, min, max, q01, q99
        Each stat is a 2D list: [chunk_size, action_dim]
    """
    stat_names = ["mean", "std", "min", "max", "q01", "q99"]
    
    # Initialize arrays for full action dimension, per position
    # Shape: [chunk_size, action_dim]
    merged = {
        stat_name: np.zeros((action_chunk_size, action_dim))
        for stat_name in stat_names
    }
    
    # Fill in EEF statistics at their correct indices, per position
    for pos in range(action_chunk_size):
        eef_stats_at_pos = eef_stats_per_position[pos]
        for key in eef_keys:
            start, end, _ = action_indices[key]
            for stat_name in stat_names:
                merged[stat_name][pos, start:end] = np.array(eef_stats_at_pos[key][stat_name])
    
    # Fill in non-EEF statistics at their correct indices
    # Non-EEF stats are the SAME for all positions (not position-dependent)
    for key in non_eef_keys:
        start, end, _ = action_indices[key]
        for stat_name in stat_names:
            non_eef_value = np.array(non_eef_stats[key][stat_name])
            for pos in range(action_chunk_size):
                merged[stat_name][pos, start:end] = non_eef_value
    
    # Convert to nested lists for JSON serialization
    return {k: v.tolist() for k, v in merged.items()}


# ============================================================================
# Main Statistics Calculation Function
# ============================================================================


def calculate_relative_dataset_statistics(
    parquet_paths: list[Path],
    modality_meta: Any,
    relative_type: str,
    frame_type: str,
    action_chunk_size: int = 50,
    random_seed: int = 42,
    num_workers: int = None,
    reservoir_size: int = 50000,
    eef_keys: Optional[set] = None,
) -> dict:
    """Calculate dataset statistics with relative action transformations.
    
    This function computes normalization statistics for a dataset where
    EEF (end-effector) actions are transformed to relative representations.
    It uses multiprocessing and streaming statistics for efficient computation
    on large datasets without loading all data into memory.
    
    Key Features:
        - Parallel episode processing with multiprocessing
        - Streaming statistics (Welford's algorithm for mean/std)
        - Reservoir sampling for quantile estimation
        - Memory-efficient: doesn't gather all data before computing
    
    Statistics Computation Strategy:
        - For each episode, randomly sample chunk starting positions
        - Within each chunk, the reference frame (S0) is the state at chunk start
        - EEF keys are transformed before computing statistics
        - Non-EEF keys use raw data (same as standard statistics)
    
    Args:
        parquet_paths: List of parquet file paths containing episode data
        modality_meta: LeRobot modality metadata with start/end indices
        relative_type: "relative" (all to S0) or "delta" (to previous)
        frame_type: "local" (reference frame) or "global" (world frame)
        action_chunk_size: Size of action chunks (should match training config)
        random_seed: Random seed for reproducibility
        num_workers: Number of parallel workers (default: cpu_count() - 1)
        reservoir_size: Size of reservoir for quantile estimation
        eef_keys: Optional explicit set of action sub-key names (without the
            "action." prefix) to transform to the relative representation. When
            None (default) the keys are auto-detected via ``_is_eef_key`` (name
            heuristic recognising ``_position``/``_pos``/``_rotation``/
            ``_rotation_6d``/``_orientation_6d``). Pass this explicitly when the
            rotation keys use a suffix the heuristic does not recognise (e.g.
            ``_6d``) so the computed statistics match the transform's ``apply_to``.
        
    Returns:
        Statistics dictionary with same format as calculate_dataset_statistics,
        plus "_relative_info" metadata field describing the transform config
    """
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    print(f"Calculating {relative_type}_{frame_type} statistics with chunk_size={action_chunk_size}...")
    print(f"Using {num_workers} workers for parallel processing")
    
    # Set master random seed for reproducibility
    np.random.seed(random_seed)
    
    # Extract sub-key indices from modality metadata
    action_indices = _get_subkey_indices(modality_meta, "action")
    state_indices = _get_subkey_indices(modality_meta, "state")
    
    # Separate EEF keys (transformed) from non-EEF keys (passed through).
    # If an explicit set is provided, trust it (validating against the modality
    # metadata); otherwise fall back to the name heuristic.
    if eef_keys is None:
        eef_keys = {k for k in action_indices.keys() if _is_eef_key(k)}
    else:
        unknown = set(eef_keys) - set(action_indices.keys())
        assert not unknown, (
            f"eef_keys not present in action modality metadata: {sorted(unknown)}. "
            f"Available: {sorted(action_indices.keys())}"
        )
        eef_keys = {k for k in eef_keys}
    non_eef_keys = {k for k in action_indices.keys() if k not in eef_keys}
    print(f"EEF keys to transform: {eef_keys}")
    print(f"Non-EEF keys (unchanged): {non_eef_keys}")
    
    # Load all parquet files in parallel. LeRobot exports are one-episode-per-
    # parquet, but the per-file ``episode_index`` restarts at 0 for every dataset
    # — so when this function is called with parquets from multiple datasets
    # (e.g. all datasets sharing one embodiment tag in precompute_dataset_stats),
    # the raw ``episode_index`` column is not unique. Always overwrite it with
    # the per-file enumeration index so the groupby below treats each parquet
    # as its own episode.
    n_threads = max(1, cpu_count())
    sorted_parquet_paths = sorted(parquet_paths)

    def _read_parquet_with_episode_index(index_and_path: tuple[int, Path]) -> pd.DataFrame:
        file_episode_index, parquet_path = index_and_path
        df = pd.read_parquet(parquet_path).copy()
        df["episode_index"] = file_episode_index
        return df

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        all_dfs = list(tqdm(
            executor.map(_read_parquet_with_episode_index, enumerate(sorted_parquet_paths)),
            total=len(sorted_parquet_paths),
            desc=f"Loading parquet files ({n_threads} threads)",
        ))
    all_data = pd.concat(all_dfs, axis=0)
    
    # Detect single-column vs multi-column layout and build remapped indices
    action_col_names = set(v[2] for v in action_indices.values())
    state_col_names = set(v[2] for v in state_indices.values())
    all_modality_cols = action_col_names | state_col_names

    remapped_action_indices = _remap_indices(action_indices)
    remapped_state_indices = _remap_indices(state_indices)

    for col in all_modality_cols:
        assert col in all_data.columns, \
            f"Column '{col}' not found. Available: {list(all_data.columns)}"
    assert 'episode_index' in all_data.columns, \
        "Column 'episode_index' not found in data after parquet loading"

    # Group by episode and prepare data for parallel processing. The
    # episode_index column above was rewritten to a per-parquet-file enumeration
    # index, so every group corresponds to exactly one source parquet file.
    print("Preparing episode data for parallel processing...")
    episode_data_list = []
    for episode_idx, ep_data in all_data.groupby('episode_index'):
        ep_data = ep_data.reset_index(drop=True)
        ep_len = len(ep_data)
        if ep_len < 2:
            continue

        action_array, _ = _build_combined_array_for_episode(ep_data, action_indices)
        state_array, _ = _build_combined_array_for_episode(ep_data, state_indices)
        episode_data_list.append((episode_idx, action_array, state_array))
    
    print(f"Total episodes to process: {len(episode_data_list)}")
    
    # Distribute episodes across workers
    num_episodes = len(episode_data_list)
    episodes_per_worker = max(1, num_episodes // num_workers)
    
    worker_batches = []
    for i in range(num_workers):
        start_idx = i * episodes_per_worker
        if i == num_workers - 1:
            end_idx = num_episodes  # Last worker gets remaining episodes
        else:
            end_idx = (i + 1) * episodes_per_worker
        
        if start_idx >= num_episodes:
            break
            
        batch = episode_data_list[start_idx:end_idx]
        if len(batch) == 0:
            continue
            
        config = {
            'action_indices': remapped_action_indices,
            'state_indices': remapped_state_indices,
            'eef_keys': eef_keys,
            'non_eef_keys': non_eef_keys,
            'relative_type': relative_type,
            'frame_type': frame_type,
            'action_chunk_size': action_chunk_size,
            'worker_seed': random_seed + i,  # Different seed per worker
            'reservoir_size': reservoir_size // num_workers,
        }
        worker_batches.append((batch, config))
    
    # Process batches in parallel
    print(f"Processing {len(worker_batches)} batches in parallel...")
    
    if num_workers > 1 and len(worker_batches) > 1:
        with Pool(processes=min(num_workers, len(worker_batches))) as pool:
            results = list(tqdm(
                pool.imap(_process_episode_batch, worker_batches),
                total=len(worker_batches),
                desc="Processing episode batches"
            ))
    else:
        # Single worker - avoid multiprocessing overhead
        results = [_process_episode_batch(batch) for batch in tqdm(worker_batches, desc="Processing episodes")]
    
    # Merge results from all workers
    print("Merging statistics from workers...")
    
    eef_dim = sum(remapped_action_indices[k][1] - remapped_action_indices[k][0] for k in sorted(eef_keys))

    # Initialize per-position streaming stats for EEF data
    merged_eef_stats_per_position = [
        StreamingStats(dim=eef_dim, reservoir_size=reservoir_size // action_chunk_size)
        for _ in range(action_chunk_size)
    ]

    merged_non_eef_stats = {}
    for key in non_eef_keys:
        start, end, _ = remapped_action_indices[key]
        merged_non_eef_stats[key] = StreamingStats(dim=end - start, reservoir_size=reservoir_size)
    
    total_chunks = 0
    
    for result in results:
        # Merge EEF stats per position
        for pos, stats_data in enumerate(result['eef_stats_per_position']):
            worker_eef = _deserialize_streaming_stats(stats_data)
            merged_eef_stats_per_position[pos].merge(worker_eef)
        
        # Merge non-EEF stats
        for key, stats_data in result['non_eef_stats'].items():
            worker_stats = _deserialize_streaming_stats(stats_data)
            merged_non_eef_stats[key].merge(worker_stats)
        
        total_chunks += result['total_chunks']
    
    print(f"Total chunks sampled: {total_chunks}")
    
    # Convert streaming stats to final statistics format
    stats = {}
    
    # Build EEF statistics by extracting each key's portion from merged stats, per position
    eef_stats_per_position = []
    for pos in range(action_chunk_size):
        merged_stats = merged_eef_stats_per_position[pos]
        eef_stats_at_pos = {}
        offset = 0
        for key in sorted(eef_keys):
            start, end, _ = remapped_action_indices[key]
            dim = end - start

            eef_stats_at_pos[key] = {
                "mean": merged_stats.mean[offset:offset + dim].tolist(),
                "std": merged_stats.std[offset:offset + dim].tolist(),
                "min": merged_stats.min_val[offset:offset + dim].tolist(),
                "max": merged_stats.max_val[offset:offset + dim].tolist(),
                "q01": merged_stats.quantile(0.01)[offset:offset + dim].tolist(),
                "q99": merged_stats.quantile(0.99)[offset:offset + dim].tolist(),
            }
            offset += dim
        eef_stats_per_position.append(eef_stats_at_pos)

    # Build non-EEF statistics from their individual streaming stats
    non_eef_stats_dict = {}
    for key in non_eef_keys:
        non_eef_stats_dict[key] = merged_non_eef_stats[key].to_stats_dict()

    # Merge into full action statistics with per-position format (combined vector)
    action_dim_combined = sum(v[1] - v[0] for v in remapped_action_indices.values())
    combined_action_stats = _merge_action_statistics_per_position(
        action_dim_combined, action_chunk_size, remapped_action_indices, eef_keys, non_eef_keys,
        eef_stats_per_position, non_eef_stats_dict
    )

    # Output action stats: always prefix with _action_stats/ to avoid
    # collisions when action and state sub-keys share the same original_key.
    # Always un-remap via _split_stats_to_columns so indices match modality.json.
    per_col_action = _split_stats_to_columns(
        combined_action_stats, remapped_action_indices, action_indices
    )
    for col, col_stats in per_col_action.items():
        stats[f"_action_stats/{col}"] = col_stats

    # Compute state statistics using combined array approach (handles multi-column)
    state_dim_combined = sum(v[1] - v[0] for v in remapped_state_indices.values())
    state_streaming = StreamingStats(dim=state_dim_combined, reservoir_size=reservoir_size)

    # Find other columns to compute statistics for (exclude all modality columns)
    other_columns = [
        col for col in all_data.columns
        if col not in all_modality_cols and col != 'episode_index'
        and not col.startswith("annotation.")
    ]
    other_streaming = {}

    # Process in batches for memory efficiency
    batch_size = 10000
    total_rows = len(all_data)

    for start_idx in range(0, total_rows, batch_size):
        end_idx = min(start_idx + batch_size, total_rows)
        batch_data = all_data.iloc[start_idx:end_idx]

        # State data - build combined array from potentially multiple columns
        state_combined, _ = _build_combined_array_for_episode(batch_data, state_indices)
        state_streaming.update_batch(state_combined)

        # Other columns
        for col in other_columns:
            if col not in other_streaming:
                first_val = batch_data[col].iloc[0]
                col_dim = np.asarray(first_val, dtype=np.float32).reshape(-1).shape[0]
                other_streaming[col] = StreamingStats(dim=col_dim, reservoir_size=reservoir_size)

            col_batch = np.vstack([
                np.asarray(x, dtype=np.float32).reshape(1, -1) for x in batch_data[col]
            ])
            other_streaming[col].update_batch(col_batch)

    # Output state stats: always prefix with _state_stats/ to avoid collisions.
    # Always un-remap via _split_stats_to_columns so indices match modality.json.
    combined_state_stats = state_streaming.to_stats_dict()
    per_col_state = _split_stats_to_columns(
        combined_state_stats, remapped_state_indices, state_indices
    )
    for col, col_stats in per_col_state.items():
        stats[f"_state_stats/{col}"] = col_stats

    for col in other_columns:
        if col in other_streaming:
            stats[col] = other_streaming[col].to_stats_dict()
    
    # Add metadata about the relative transform configuration
    stats["_relative_info"] = {
        "relative_type": relative_type,
        "frame_type": frame_type,
        "action_chunk_size": action_chunk_size,
        "transformed_keys": sorted(eef_keys),
        "non_transformed_keys": sorted(non_eef_keys),
        "total_chunks_sampled": total_chunks,
        "random_seed": random_seed,
        "num_workers": num_workers,
        "reservoir_size": reservoir_size,
        "description": (
            f"EEF keys {sorted(eef_keys)} are in {relative_type}_{frame_type} representation. "
            f"Statistics are PER POSITION in the chunk (shape [chunk_size={action_chunk_size}, dim]). "
            f"Non-EEF keys {sorted(non_eef_keys)} are repeated for each position. "
            f"Computed with {num_workers} parallel workers using streaming statistics."
        ),
    }
    
    return stats


# ============================================================================
# RelativeActionTransform Class
# ============================================================================


class RelativeActionTransform(InvertibleModalityTransform):
    """Transform absolute actions to relative representations.
    
    This transform converts absolute end-effector (EEF) poses to relative
    representations, which can improve generalization across different
    initial robot configurations. It is invertible, meaning the original
    absolute actions can be recovered.
    
    Transform Pipeline Position:
        This transform should be placed BEFORE StateActionTransform (normalization)
        but AFTER StateActionToTensor:
        
        StateActionToTensor → RelativeActionTransform → StateActionTransform → ConcatTransform
    
    Attributes:
        apply_to: List of action keys to transform (e.g., ["action.leftHand_position"])
        state_keys_mapping: Dict mapping action keys to corresponding state keys
        relative_type: "relative" (all to S0) or "delta" (to previous)
        frame_type: "local" (reference frame) or "global" (world frame)
    
    Example:
        >>> transform = RelativeActionTransform(
        ...     apply_to=["action.leftHand_position", "action.leftHand_6d"],
        ...     state_keys_mapping={
        ...         "action.leftHand_position": "state.leftHand_position",
        ...         "action.leftHand_6d": "state.leftHand_6d",
        ...     },
        ...     relative_type="delta",
        ...     frame_type="local",
        ... )
        >>> data = transform.apply(data)  # Convert to relative
        >>> data = transform.unapply(data)  # Convert back to absolute
    """
    
    apply_to: list[str] = Field(
        ..., 
        description="Action keys to transform to relative representation."
    )
    state_keys_mapping: dict[str, str] = Field(
        ..., 
        description="Mapping from action key to corresponding state key."
    )
    relative_type: Literal["relative", "delta"] = Field(
        default="relative",
        description="Type of relative representation: 'relative' (to S0) or 'delta' (to previous)."
    )
    frame_type: Literal["local", "global"] = Field(
        default="local",
        description="Frame type: 'local' (reference frame) or 'global' (world frame)."
    )
    action_chunk_size: int = Field(
        default=50,
        description="Action chunk size for computing relative statistics. Default is 50."
    )
    
    def model_dump(self, *args, **kwargs):
        """Custom serialization to include only relevant fields for JSON mode."""
        if kwargs.get("mode", "python") == "json":
            include = {"apply_to", "state_keys_mapping", "relative_type", "frame_type", "action_chunk_size"}
        else:
            include = kwargs.pop("include", None)
        return super().model_dump(*args, include=include, **kwargs)
    
    def is_rotation_key(self, key: str) -> bool:
        """Check if a key represents rotation data."""
        return _is_rotation_key(key)
    
    def is_position_key(self, key: str) -> bool:
        """Check if a key represents position data."""
        return _is_position_key(key)
    
    # ========================================
    # Rotation/Position Conversion Methods
    # ========================================
    
    def _rot6d_to_matrix(self, rot6d: torch.Tensor) -> torch.Tensor:
        """Convert 6D rotation representation to rotation matrix.
        
        Args:
            rot6d: Tensor of shape [..., 6]
            
        Returns:
            Tensor of shape [..., 3, 3]
        """
        return pt.rotation_6d_to_matrix(rot6d)
    
    def _matrix_to_rot6d(self, matrix: torch.Tensor) -> torch.Tensor:
        """Convert rotation matrix to 6D rotation representation.
        
        Args:
            matrix: Tensor of shape [..., 3, 3]
            
        Returns:
            Tensor of shape [..., 6]
        """
        return pt.matrix_to_rotation_6d(matrix)
    
    # ========================================
    # Relative Transform Computation Methods
    # ========================================
    
    def _compute_relative_rotation(
        self, 
        R_curr: torch.Tensor, 
        R_ref: torch.Tensor
    ) -> torch.Tensor:
        """Compute relative rotation based on frame type.
        
        Args:
            R_curr: Current rotation matrix [..., 3, 3]
            R_ref: Reference rotation matrix [..., 3, 3]
            
        Returns:
            Relative rotation matrix [..., 3, 3]
        """
        if self.frame_type == "local":
            # Local frame: delta_R = R_ref.T @ R_curr
            return torch.matmul(R_ref.transpose(-2, -1), R_curr)
        else:
            # Global frame: delta_R = R_curr @ R_ref.T
            return torch.matmul(R_curr, R_ref.transpose(-2, -1))
    
    def _compute_absolute_rotation(
        self,
        delta_R: torch.Tensor,
        R_ref: torch.Tensor
    ) -> torch.Tensor:
        """Compute absolute rotation from relative rotation (inverse of _compute_relative_rotation).
        
        Args:
            delta_R: Relative rotation matrix [..., 3, 3]
            R_ref: Reference rotation matrix [..., 3, 3]
            
        Returns:
            Absolute rotation matrix [..., 3, 3]
        """
        if self.frame_type == "local":
            # Inverse of: delta_R = R_ref.T @ R_curr → R_curr = R_ref @ delta_R
            return torch.matmul(R_ref, delta_R)
        else:
            # Inverse of: delta_R = R_curr @ R_ref.T → R_curr = delta_R @ R_ref
            return torch.matmul(delta_R, R_ref)
    
    def _compute_relative_position(
        self,
        pos_curr: torch.Tensor,
        pos_ref: torch.Tensor,
        R_ref: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute relative position based on frame type.
        
        Args:
            pos_curr: Current position [..., 3]
            pos_ref: Reference position [..., 3]
            R_ref: Reference rotation matrix [..., 3, 3] (required for local frame)
            
        Returns:
            Relative position [..., 3]
        """
        delta_pos = pos_curr - pos_ref
        
        if self.frame_type == "local" and R_ref is not None:
            # Local frame: rotate delta into reference frame coordinates
            delta_pos_expanded = delta_pos.unsqueeze(-1)  # [..., 3, 1]
            R_ref_T = R_ref.transpose(-2, -1)  # [..., 3, 3]
            result = torch.matmul(R_ref_T, delta_pos_expanded)  # [..., 3, 1]
            return result.squeeze(-1)  # [..., 3]
        else:
            # Global frame: delta in world coordinates
            return delta_pos
    
    def _compute_absolute_position(
        self,
        delta_pos: torch.Tensor,
        pos_ref: torch.Tensor,
        R_ref: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute absolute position from relative position (inverse of _compute_relative_position).
        
        Args:
            delta_pos: Relative position [..., 3]
            pos_ref: Reference position [..., 3]
            R_ref: Reference rotation matrix [..., 3, 3] (required for local frame)
            
        Returns:
            Absolute position [..., 3]
        """
        if self.frame_type == "local" and R_ref is not None:
            # Inverse of local frame transform: rotate back to world coordinates
            delta_pos_expanded = delta_pos.unsqueeze(-1)  # [..., 3, 1]
            result = torch.matmul(R_ref, delta_pos_expanded)  # [..., 3, 1]
            return result.squeeze(-1) + pos_ref  # [..., 3]
        else:
            # Inverse of global frame: simple addition
            return delta_pos + pos_ref
    
    def _get_rotation_key_for_position(self, position_key: str) -> Optional[str]:
        """Get the corresponding rotation key for a position key.
        
        For local frame transforms, positions need the corresponding rotation
        to transform into the reference frame.
        
        Args:
            position_key: Position action key (e.g., "action.leftHand_position")
            
        Returns:
            Matching rotation key if found, None otherwise
        """
        if not self.is_position_key(position_key):
            return None
        
        if position_key.endswith("_position"):
            prefix = position_key.removesuffix("_position")
        elif position_key.endswith("_pos"):
            prefix = position_key.removesuffix("_pos")
        else:
            return None
        
        rotation_candidates = [
            f"{prefix}_rotation_6d",
            f"{prefix}_orientation_6d",
            f"{prefix}_6d",
            f"{prefix}_rotation",
        ]
        for rotation_key in rotation_candidates:
            if rotation_key in self.apply_to:
                return rotation_key
        
        return None
    
    # ========================================
    # Apply Methods (Absolute → Relative)
    # ========================================
    
    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        """Convert absolute actions to relative representation.
        
        The reference state is taken from the first timestep (index 0) of the
        state data.
        
        During inference (self.training=False), this method is a no-op since
        action keys are not present in the input data - they are being predicted.
        Use unapply() to convert relative predictions back to absolute.
        
        Args:
            data: Dictionary containing action and state tensors in chunk
            
        Returns:
            Modified data dictionary with relative actions (training mode),
            or unchanged data (inference mode)
        """
        # During inference, action keys don't exist in input data.
        # Skip the transform - unapply() will be used to convert predictions back.
        if not self.training:
            return data
        
        # First pass: collect all inputs and reference states
        # We need to read all absolute action data before modifying any
        action_inputs = {}  # action_key -> (action_data, ref_state, ref_matrix)
        
        for action_key in self.apply_to:
            state_key = self.state_keys_mapping[action_key]
            if action_key not in data or state_key not in data:
                raise ValueError(f"Action key '{action_key}' or state key '{state_key}' not found in data."
                                 f"Cannot apply relative transform. Available keys: {list(data.keys())}.")

            action_data = data[action_key].clone()  # Clone to avoid issues
            state_data = data[state_key]
            
            if action_data.shape[0] != self.action_chunk_size:
                raise ValueError(
                    f"Action chunk size mismatch for key '{action_key}': "
                    f"got {action_data.shape[0]}, expected {self.action_chunk_size}. "
                )
            
            # Get reference state (first timestep)
            ref_state = state_data[0]
            
            # For rotation keys, also compute the rotation matrix
            ref_matrix = None
            if self.is_rotation_key(action_key):
                ref_matrix = self._rot6d_to_matrix(ref_state)
            
            action_inputs[action_key] = (action_data, ref_state, ref_matrix)
        
        # Second pass: compute relative values
        results = {}
        
        for action_key, (action_data, ref_state, ref_matrix) in action_inputs.items():
            if self.is_rotation_key(action_key):
                results[action_key] = self._apply_rotation(action_data, ref_state, ref_matrix)
            elif self.is_position_key(action_key):
                # Get corresponding rotation data for local frame transform
                rot_key = self._get_rotation_key_for_position(action_key)
                ref_rotation = None
                action_rotation_data = None
                
                if rot_key and rot_key in action_inputs:
                    _, _, ref_rotation = action_inputs[rot_key]
                    action_rotation_data = action_inputs[rot_key][0]
                
                results[action_key] = self._apply_position(
                    action_data, ref_state, ref_rotation, action_rotation_data
                )
        
        # Update data dictionary with results
        for action_key, result in results.items():
            data[action_key] = result
        
        return data
    
    def _apply_rotation(
        self,
        action_rot6d: torch.Tensor,
        ref_rot6d: torch.Tensor,
        ref_matrix: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply relative transform to rotation data.
        
        Args:
            action_rot6d: Action rotation data [T, 6]
            ref_rot6d: Reference rotation [6]
            ref_matrix: Pre-computed reference rotation matrix [3, 3]
            
        Returns:
            Relative rotation data [T, 6]
        """
        T = action_rot6d.shape[0]
        
        # Convert to matrices
        action_matrices = self._rot6d_to_matrix(action_rot6d)  # [T, 3, 3]
        if ref_matrix is None:
            ref_matrix = self._rot6d_to_matrix(ref_rot6d)  # [3, 3]
        
        if self.relative_type == "relative":
            # All actions relative to S0
            ref_expanded = ref_matrix.unsqueeze(0).expand(T, -1, -1)  # [T, 3, 3]
            relative_matrices = self._compute_relative_rotation(action_matrices, ref_expanded)
        else:
            # Delta: each action relative to previous pose
            relative_matrices = torch.zeros_like(action_matrices)
            prev_matrix = ref_matrix
            
            for t in range(T):
                curr_matrix = action_matrices[t]
                relative_matrices[t] = self._compute_relative_rotation(curr_matrix, prev_matrix)
                prev_matrix = curr_matrix
        
        return self._matrix_to_rot6d(relative_matrices)  # [T, 6]
    
    def _apply_position(
        self,
        action_pos: torch.Tensor,
        ref_pos: torch.Tensor,
        ref_rotation: Optional[torch.Tensor] = None,
        action_rotation_data: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply relative transform to position data.
        
        Args:
            action_pos: Action position data [T, 3]
            ref_pos: Reference position [3]
            ref_rotation: Reference rotation matrix [3, 3] (S0 rotation, for local frame)
            action_rotation_data: Absolute action rotation data [T, 6] (for delta+local mode)
            
        Returns:
            Relative position data [T, 3]
        """
        T = action_pos.shape[0]
        
        if self.relative_type == "relative":
            # All positions relative to S0
            ref_expanded = ref_pos.unsqueeze(0).expand(T, -1)  # [T, 3]
            if ref_rotation is not None and self.frame_type == "local":
                ref_rot_expanded = ref_rotation.unsqueeze(0).expand(T, -1, -1)  # [T, 3, 3]
            else:
                ref_rot_expanded = None
            return self._compute_relative_position(action_pos, ref_expanded, ref_rot_expanded)
        else:
            # Delta: each position relative to previous
            relative_pos = torch.zeros_like(action_pos)
            prev_pos = ref_pos
            prev_rot = ref_rotation
            
            # For delta+local, need rotation at each timestep
            action_rot_matrices = None
            if action_rotation_data is not None and self.frame_type == "local":
                action_rot_matrices = self._rot6d_to_matrix(action_rotation_data)  # [T, 3, 3]
            
            for t in range(T):
                curr_pos = action_pos[t]
                relative_pos[t] = self._compute_relative_position(curr_pos, prev_pos, prev_rot)
                prev_pos = curr_pos
                # Update reference rotation for next iteration
                if action_rot_matrices is not None:
                    prev_rot = action_rot_matrices[t]
            
            return relative_pos
    
    # ========================================
    # Unapply Methods (Relative → Absolute)
    # ========================================
    
    def unapply(self, data: dict[str, Any]) -> dict[str, Any]:
        """Convert relative actions back to absolute representation.
        
        Supports both single-sample ([T, dim]) and batched ([B, T, dim]) inputs.
        
        Note: The reference states must be available in the data dict under state keys.
        Pass state data to unapply() along with action data.
        
        Args:
            data: Dictionary containing relative action tensors and state tensors
            
        Returns:
            Modified data dictionary with absolute actions
        """
        sample_key = next((k for k in self.apply_to if k in data), None)
        if sample_key is not None and data[sample_key].dim() == 3:
            return self._unapply_batched(data)
        return self._unapply_single(data)

    def _unapply_batched(self, data: dict[str, Any]) -> dict[str, Any]:
        """Handle batched unapply by iterating over the batch dimension."""
        sample_key = next(k for k in self.apply_to if k in data)
        B = data[sample_key].shape[0]

        per_sample = []
        for i in range(B):
            single = {
                k: (v[i] if isinstance(v, torch.Tensor) and v.dim() >= 2 else v)
                for k, v in data.items()
            }
            per_sample.append(self._unapply_single(single))

        for k in data:
            if isinstance(data[k], torch.Tensor) and data[k].dim() >= 2:
                data[k] = torch.stack([s[k] for s in per_sample])
        return data

    def _unapply_single(self, data: dict[str, Any]) -> dict[str, Any]:
        """Convert relative actions back to absolute for a single sample."""
        # First pass: recover rotations (needed for local frame position recovery)
        recovered_rot6d = {}
        recovered_rot_matrices = {}
        
        for action_key in self.apply_to:
            if not self.is_rotation_key(action_key):
                continue
            
            if action_key not in data:
                continue
            
            # Get reference from state key in data
            state_key = self.state_keys_mapping.get(action_key)
            if state_key is None or state_key not in data:
                raise ValueError(
                    f"State key '{state_key}' required for action key '{action_key}'. "
                    f"Pass state data to unapply(). Available keys: {list(data.keys())}"
                )
            state_data = data[state_key]
            # Extract first element and squeeze out all leading dimensions to get [6]
            ref_rot6d = state_data
            while ref_rot6d.dim() > 1:
                ref_rot6d = ref_rot6d[0]
            
            ref_matrix = self._rot6d_to_matrix(ref_rot6d)
            
            relative_rot6d = data[action_key]
            abs_rot6d, abs_matrices = self._unapply_rotation(
                relative_rot6d, ref_rot6d, ref_matrix, return_matrices=True
            )
            recovered_rot6d[action_key] = abs_rot6d
            recovered_rot_matrices[action_key] = abs_matrices
        
        # Second pass: recover positions
        recovered_positions = {}
        
        for action_key in self.apply_to:
            if not self.is_position_key(action_key):
                continue
            
            if action_key not in data:
                continue
            
            # Get reference from state key in data
            state_key = self.state_keys_mapping.get(action_key)
            if state_key is None or state_key not in data:
                raise ValueError(
                    f"State key '{state_key}' required for action key '{action_key}'. "
                    f"Pass state data to unapply(). Available keys: {list(data.keys())}"
                )
            state_data = data[state_key]
            # Extract first element and squeeze out all leading dimensions to get [3]
            ref_pos = state_data
            while ref_pos.dim() > 1:
                ref_pos = ref_pos[0]
            
            # Get corresponding rotation for local frame
            rot_key = self._get_rotation_key_for_position(action_key)
            ref_rotation = None
            abs_rot_matrices = None
            
            if rot_key:
                # Compute ref_rotation from state key
                rot_state_key = self.state_keys_mapping.get(rot_key)
                if rot_state_key and rot_state_key in data:
                    rot_state_data = data[rot_state_key]
                    # Extract first element and squeeze out all leading dimensions to get [6]
                    ref_rot6d_for_pos = rot_state_data
                    while ref_rot6d_for_pos.dim() > 1:
                        ref_rot6d_for_pos = ref_rot6d_for_pos[0]
                    ref_rotation = self._rot6d_to_matrix(ref_rot6d_for_pos)
                if rot_key in recovered_rot_matrices:
                    abs_rot_matrices = recovered_rot_matrices[rot_key]
            
            relative_pos = data[action_key]
            recovered_positions[action_key] = self._unapply_position(
                relative_pos, ref_pos, ref_rotation, abs_rot_matrices
            )
        
        # Update data dictionary with recovered values
        for action_key, value in recovered_rot6d.items():
            data[action_key] = value
        for action_key, value in recovered_positions.items():
            data[action_key] = value
        
        return data
    
    def _unapply_rotation(
        self,
        relative_rot6d: torch.Tensor,
        ref_rot6d: torch.Tensor,
        ref_matrix: Optional[torch.Tensor] = None,
        return_matrices: bool = False
    ):
        """Recover absolute rotation from relative rotation.
        
        Args:
            relative_rot6d: Relative rotation data [T, 6]
            ref_rot6d: Reference rotation [6]
            ref_matrix: Pre-computed reference rotation matrix [3, 3]
            return_matrices: If True, also return the absolute rotation matrices
            
        Returns:
            Absolute rotation data [T, 6], and optionally the matrices [T, 3, 3]
        """
        T = relative_rot6d.shape[0]
        
        # Convert to matrices
        relative_matrices = self._rot6d_to_matrix(relative_rot6d)  # [T, 3, 3]
        if ref_matrix is None:
            ref_matrix = self._rot6d_to_matrix(ref_rot6d)  # [3, 3]
        
        # Ensure ref_matrix has same dtype and device as relative_matrices
        ref_matrix = ref_matrix.to(dtype=relative_matrices.dtype, device=relative_matrices.device)
        
        if self.relative_type == "relative":
            # All actions were relative to S0
            ref_expanded = ref_matrix.unsqueeze(0).expand(T, -1, -1)  # [T, 3, 3]
            absolute_matrices = self._compute_absolute_rotation(relative_matrices, ref_expanded)
        else:
            # Delta: recover sequentially
            absolute_matrices = torch.zeros_like(relative_matrices)
            prev_matrix = ref_matrix
            
            for t in range(T):
                delta_matrix = relative_matrices[t]
                curr_matrix = self._compute_absolute_rotation(delta_matrix, prev_matrix)
                absolute_matrices[t] = curr_matrix
                prev_matrix = curr_matrix
        
        result = self._matrix_to_rot6d(absolute_matrices)  # [T, 6]
        
        if return_matrices:
            return result, absolute_matrices
        return result
    
    def _unapply_position(
        self,
        relative_pos: torch.Tensor,
        ref_pos: torch.Tensor,
        ref_rotation: Optional[torch.Tensor] = None,
        abs_rot_matrices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Recover absolute position from relative position.
        
        Args:
            relative_pos: Relative position data [T, 3]
            ref_pos: Reference position [3]
            ref_rotation: Reference rotation matrix [3, 3] (S0 rotation, for local frame)
            abs_rot_matrices: Absolute rotation matrices [T, 3, 3] (for delta+local mode)
            
        Returns:
            Absolute position data [T, 3]
        """
        T = relative_pos.shape[0]
        
        # Ensure ref tensors have same dtype and device as relative_pos
        ref_pos = ref_pos.to(dtype=relative_pos.dtype, device=relative_pos.device)
        if ref_rotation is not None:
            ref_rotation = ref_rotation.to(dtype=relative_pos.dtype, device=relative_pos.device)
        
        if self.relative_type == "relative":
            # All positions were relative to S0
            ref_expanded = ref_pos.unsqueeze(0).expand(T, -1)  # [T, 3]
            if ref_rotation is not None and self.frame_type == "local":
                ref_rot_expanded = ref_rotation.unsqueeze(0).expand(T, -1, -1)  # [T, 3, 3]
            else:
                ref_rot_expanded = None
            return self._compute_absolute_position(relative_pos, ref_expanded, ref_rot_expanded)
        else:
            # Delta: recover sequentially
            absolute_pos = torch.zeros_like(relative_pos)
            prev_pos = ref_pos
            prev_rot = ref_rotation
            
            for t in range(T):
                delta_pos = relative_pos[t]
                curr_pos = self._compute_absolute_position(delta_pos, prev_pos, prev_rot)
                absolute_pos[t] = curr_pos
                prev_pos = curr_pos
                # Update reference rotation for next iteration
                if abs_rot_matrices is not None and self.frame_type == "local":
                    prev_rot = abs_rot_matrices[t]
            
            return absolute_pos
