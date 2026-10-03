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
from pydantic import Field, PrivateAttr

from ..schema import DatasetMetadata, StateActionMetadata
from .base import InvertibleModalityTransform


class ConcatTransform(InvertibleModalityTransform):
    """
    Concatenate the keys according to specified order.
    """

    # -- We inherit from ModalityTransform, so we keep apply_to as well --
    apply_to: list[str] = Field(
        default_factory=list, description="Not used in this transform, kept for compatibility."
    )

    video_concat_order: Optional[list[str]] = Field(
        default=None,
        description="Concatenation order for each video modality. "
        "Format: ['video.ego_view_pad_res224_freq20', ...]. "
        "If None or empty, video concatenation is skipped. "
        "In training mode (self.training=True), video concatenation is also skipped "
        "to preserve individual video keys for the dataloader.",
    )

    state_concat_order: Optional[list[str]] = Field(
        default=None,
        description="Concatenation order for each state modality. "
        "Format: ['state.position', 'state.velocity', ...].",
    )

    action_concat_order: Optional[list[str]] = Field(
        default=None,
        description="Concatenation order for each action modality. "
        "Format: ['action.position', 'action.velocity', ...].",
    )

    action_dims: dict[str, int] = Field(
        default_factory=dict,
        description="The dimensions of the action keys (from metadata).",
    )
    state_dims: dict[str, int] = Field(
        default_factory=dict,
        description="The dimensions of the state keys (from metadata).",
    )
    # Actual dimensions observed during apply(), used for correct unapply().
    # Differs from state_dims when upstream transforms change dimensions (e.g., sin-cos doubles dims).
    _applied_state_dims: dict[str, int] = PrivateAttr(default_factory=dict)

    def model_dump(self, *args, **kwargs):
        if kwargs.get("mode", "python") == "json":
            include = {
                "apply_to",
                "video_concat_order",
                "state_concat_order",
                "action_concat_order",
            }
        else:
            include = kwargs.pop("include", None)

        return super().model_dump(*args, include=include, **kwargs)

    def apply(self, data: dict) -> dict:
        grouped_keys = {}
        for key in data.keys():
            try:
                modality, _ = key.split(".")
            except:  # noqa: E722
                ### Handle language annotation special case
                if "annotation" in key:
                    modality = "language"
                else:
                    modality = "others"
            if modality not in grouped_keys:
                grouped_keys[modality] = []
            grouped_keys[modality].append(key)

        # Video concatenation: only perform in eval mode (self.training=False)
        # In training mode, individual video keys are preserved for the dataloader
        should_concat_video = "video" in grouped_keys and self.video_concat_order and not self.training
        if should_concat_video:
            # Check if keys in video_concat_order are indeed contained in the data.
            # If not, then the keys are misspecified
            video_keys = grouped_keys["video"]
            assert all(
                item in video_keys for item in self.video_concat_order
            ), f"keys in video_concat_order are misspecified, \n{video_keys=}, \n{self.video_concat_order=}"

            # Process each video view
            unsqueezed_videos = []
            for video_key in self.video_concat_order:
                video_data = data.pop(video_key)
                unsqueezed_video = np.expand_dims(
                    video_data, axis=-4
                )  # [..., H, W, C] -> [..., 1, H, W, C]
                unsqueezed_videos.append(unsqueezed_video)
            # Concatenate along the new axis
            unsqueezed_video = np.concatenate(unsqueezed_videos, axis=-4)  # [..., V, H, W, C]

            # Video
            data["video"] = unsqueezed_video

        # "state"
        if "state" in grouped_keys:
            state_keys = grouped_keys["state"]
            assert self.state_concat_order is not None, f"{self.state_concat_order=}"
            assert all(
                item in state_keys for item in self.state_concat_order
            ), f"keys in state_concat_order are misspecified, \n{state_keys=}, \n{self.state_concat_order=}"
            # Check the state dims and record ACTUAL dimensions for unapply
            state_tensors = []
            for key in self.state_concat_order:
                target_shapes = [self.state_dims[key]]
                if self.is_rotation_key(key):
                    target_shapes.append(6)  # Allow for rotation_6d
                # if key in ["state.right_arm", "state.right_hand"]:
                target_shapes.append(self.state_dims[key] * 2)  # Allow for sin-cos transform
                assert (
                    data[key].shape[-1] in target_shapes
                ), f"State dim mismatch for {key=}, {data[key].shape[-1]=}, {target_shapes=}"
                # Record actual dimension for unapply (may differ from metadata after transforms)
                self._applied_state_dims[key] = data[key].shape[-1]
                state_tensors.append(data.pop(key))
            # Concatenate the state keys
            # We'll have StateActionToTensor before this transform, so here we use torch.cat
            data["state"] = torch.cat(state_tensors, dim=-1)  # [T, D_state]

        if "action" in grouped_keys:
            action_keys = grouped_keys["action"]
            assert self.action_concat_order is not None, f"{self.action_concat_order=}"
            # Check if all keys in concat_order are present
            assert set(self.action_concat_order) == set(
                action_keys
            ), f"{set(self.action_concat_order)=}, {set(action_keys)=}"
            # Record the action dims
            for key in self.action_concat_order:
                target_shapes = [self.action_dims[key]]
                if self.is_rotation_key(key):
                    target_shapes.append(3)  # Allow for axis angle
                assert (
                    self.action_dims[key] == data[key].shape[-1]
                ), f"Action dim mismatch for {key=}, {self.action_dims[key]=}, {data[key].shape[-1]=}"
            # Concatenate the action keys
            # We'll have StateActionToTensor before this transform, so here we use torch.cat
            data["action"] = torch.cat(
                [data.pop(key) for key in self.action_concat_order], dim=-1
            )  # [T, D_action]

        return data

    def unapply(self, data: dict) -> dict:
        start_dim = 0
        assert "action" in data, f"{data.keys()=}"
        # For those dataset without actions (LAPA), we'll never run unapply
        assert self.action_concat_order is not None, f"{self.action_concat_order=}"
        action_tensor = data.pop("action")

        ##判断在inference的时候，是否需要把模型的输出，从6dPose转成roll/pitch/yaw
        dim_all = 0
        for key in self.action_concat_order:
             dim_all += self.action_dims[key]
        if dim_all != action_tensor.shape[-1]:
            use_pose = True
        else:
            use_pose = False

        for key in self.action_concat_order:
            if key not in self.action_dims:
                raise ValueError(f"Action dim {key} not found in action_dims.")
            end_dim = start_dim + self.action_dims[key]

            # Only adjust for euler keys (head/waist without _6d suffix)
            # Skip adjustment for _6d keys which already have 6 dimensions
            if use_pose and ('head' in key or 'waist' in key) and not key.endswith('_6d'):
                end_dim = end_dim + 3
                
            data[key] = action_tensor[..., start_dim:end_dim]
            start_dim = end_dim
        if "state" in data:
            assert self.state_concat_order is not None, f"{self.state_concat_order=}"
            start_dim = 0
            state_tensor = data.pop("state")
            for key in self.state_concat_order:
                # Use dims from apply() if available, else fall back to metadata
                dim = self._applied_state_dims.get(key, self.state_dims[key])
                end_dim = start_dim + dim
                data[key] = state_tensor[..., start_dim:end_dim]
                start_dim = end_dim
        return data

    def __call__(self, data: dict) -> dict:
        return self.apply(data)

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

    def is_rotation_key(self, key: str) -> bool:
        modality_config = self.get_modality_metadata(key)
        return modality_config.rotation_type is not None

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
