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

from typing import Optional

import numpy as np
import torch
from pydantic import Field

from ..schema import DatasetMetadata, StateActionMetadata
from .base import InvertibleModalityTransform


class StateActionMaskTransform(InvertibleModalityTransform):
    """
    Generate masks for concatenated state and action tensors.
    
    For action masks:
    - Gripper-related dimensions are masked out (0)
    - Other dimensions are kept (1)
    
    For state masks:
    - All dimensions are kept (1)
    
    This transform should be applied after ConcatTransform.
    """

    apply_to: list[str] = Field(
        default_factory=list, description="Not used in this transform, kept for compatibility."
    )

    state_concat_order: Optional[list[str]] = Field(
        default=None,
        description="Concatenation order for each state modality. "
        "Format: ['state.position', 'state.velocity', ...]. "
        "Must match the order used in ConcatTransform.",
    )

    action_concat_order: Optional[list[str]] = Field(
        default=None,
        description="Concatenation order for each action modality. "
        "Format: ['action.position', 'action.velocity', ...]. "
        "Must match the order used in ConcatTransform.",
    )

    max_state_dim: Optional[int] = Field(
        default=None,
        description="Maximum state dimension for padding. If None, no padding is applied.",
    )

    max_action_dim: Optional[int] = Field(
        default=None,
        description="Maximum action dimension for padding. If None, no padding is applied.",
    )

    action_dims: dict[str, int] = Field(
        default_factory=dict,
        description="The dimensions of the action keys.",
    )
    
    state_dims: dict[str, int] = Field(
        default_factory=dict,
        description="The dimensions of the state keys.",
    )

    # Track sincos transformed state keys for dimension doubling
    sincos_state_keys: set[str] = Field(
        default_factory=set,
        description="State keys that have been sin-cos transformed (dimensions are doubled).",
    )

    def get_modality_metadata(self, key: str) -> StateActionMetadata:
        modality, subkey = key.split(".")
        assert self.dataset_metadata is not None, "Metadata not set"
        modality_config = getattr(self.dataset_metadata.modalities, modality)
        assert subkey in modality_config, f"{subkey=} not found in {modality_config=}"
        assert isinstance(
            modality_config[subkey], StateActionMetadata
        ), f"Expected {StateActionMetadata} for {subkey=}, got {type(modality_config[subkey])=}"
        return modality_config[subkey]

    def get_state_action_dims(self, key: str) -> int:
        """Get the dimension of a state or action key from the dataset metadata."""
        modality_config = self.get_modality_metadata(key)
        shape = modality_config.shape
        assert len(shape) == 1, f"{shape=}"
        return shape[0]

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        """Set the metadata and compute the dimensions of the state and action keys."""
        super().set_metadata(dataset_metadata)
        # Pre-compute the dimensions of the state and action keys
        if self.action_concat_order is not None:
            for key in self.action_concat_order:
                self.action_dims[key] = self.get_state_action_dims(key)
        if self.state_concat_order is not None:
            for key in self.state_concat_order:
                self.state_dims[key] = self.get_state_action_dims(key)

    def _is_gripper_key(self, key: str) -> bool:
        """Check if a key is gripper-related."""
        return "gripper" in key.lower()

    def _compute_action_mask(self) -> np.ndarray:
        """Compute the action mask based on the concat order."""
        if self.action_concat_order is None:
            return np.array([], dtype=np.float32)
        
        mask_parts = []
        for key in self.action_concat_order:
            key_dim = self.action_dims.get(key, 0)
            is_gripper = self._is_gripper_key(key)
            # Gripper dims are masked (0), others are kept (1)
            mask_parts.extend([0.0 if is_gripper else 1.0] * key_dim)
        
        return np.asarray(mask_parts, dtype=np.float32)

    def _compute_state_mask(self) -> np.ndarray:
        """Compute the state mask based on the concat order."""
        if self.state_concat_order is None:
            return np.array([], dtype=np.float32)
        
        mask_parts = []
        for key in self.state_concat_order:
            key_dim = self.state_dims.get(key, 0)
            # Check if this key has been sin-cos transformed (doubled dimensions)
            if key in self.sincos_state_keys:
                key_dim *= 2
            # All state dims are kept (1)
            mask_parts.extend([1.0] * key_dim)
        
        return np.asarray(mask_parts, dtype=np.float32)

    def _pad_last_dim(self, x: np.ndarray, target_dim: int) -> np.ndarray:
        """Pad the last dimension of x to target_dim."""
        if x.shape[-1] == target_dim:
            return x
        if x.shape[-1] > target_dim:
            raise ValueError(f"Cannot pad: x_dim={x.shape[-1]} > target_dim={target_dim}")
        pad_width = [(0, 0)] * x.ndim
        pad_width[-1] = (0, target_dim - x.shape[-1])
        return np.pad(x, pad_width, mode="constant", constant_values=0)

    def _expand_mask(self, mask_vec: np.ndarray, time_len: int) -> np.ndarray:
        """Expand a 1D mask vector to [T, D] by broadcasting."""
        if mask_vec.ndim != 1:
            raise ValueError(f"Expected 1D mask_vec, got shape={mask_vec.shape}")
        return np.broadcast_to(mask_vec[None, :], (time_len, mask_vec.shape[0])).astype(np.float32, copy=False)

    def apply(self, data: dict) -> dict:
        """
        Generate masks for state and action tensors.
        
        Expects 'state' and 'action' keys in data (after ConcatTransform).
        Adds 'state_mask' and 'action_mask' keys.
        """
        # Generate action mask
        if "action" in data and self.action_concat_order is not None:
            action = data["action"]
            action_mask_vec = self._compute_action_mask()
            
            # Get time dimension from action tensor
            if isinstance(action, torch.Tensor):
                action_np = action.numpy()
            else:
                action_np = action
            time_len = action_np.shape[0]
            
            # Pad action and mask if max_action_dim is specified
            if self.max_action_dim is not None:
                if isinstance(action, torch.Tensor):
                    action_np = action.numpy()
                    action_np = self._pad_last_dim(action_np, self.max_action_dim)
                    data["action"] = torch.from_numpy(action_np)
                else:
                    data["action"] = self._pad_last_dim(action_np, self.max_action_dim)
                action_mask_vec = self._pad_last_dim(action_mask_vec, self.max_action_dim)
            
            # Expand to [T, D]
            action_mask = self._expand_mask(action_mask_vec, time_len)
            data["action_mask"] = action_mask

        # Generate state mask
        if "state" in data and self.state_concat_order is not None:
            state = data["state"]
            state_mask_vec = self._compute_state_mask()
            
            # Get time dimension from state tensor
            if isinstance(state, torch.Tensor):
                state_np = state.numpy()
            else:
                state_np = state
            time_len = state_np.shape[0]
            
            # Pad state and mask if max_state_dim is specified
            if self.max_state_dim is not None:
                if isinstance(state, torch.Tensor):
                    state_np = state.numpy()
                    state_np = self._pad_last_dim(state_np, self.max_state_dim)
                    data["state"] = torch.from_numpy(state_np)
                else:
                    data["state"] = self._pad_last_dim(state_np, self.max_state_dim)
                state_mask_vec = self._pad_last_dim(state_mask_vec, self.max_state_dim)
            
            # Expand to [T, D]
            state_mask = self._expand_mask(state_mask_vec, time_len)
            data["state_mask"] = state_mask

        return data

    def unapply(self, data: dict) -> dict:
        """Remove masks from data (masks are not reversible)."""
        # Just remove the mask keys if present
        data.pop("action_mask", None)
        data.pop("state_mask", None)
        return data

    def __call__(self, data: dict) -> dict:
        return self.apply(data)
