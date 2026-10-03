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

from typing import Any, Callable, ClassVar, Literal

import albumentations as A
import cv2
import numpy as np
import torch
import torchvision.transforms.v2 as T
from einops import rearrange
from pydantic import Field, PrivateAttr, field_validator
from PIL import Image

from ..schema import DatasetMetadata
from .base import ModalityTransform
from torchvision.transforms import Lambda
from torchvision.transforms import functional as F
from typing import Literal, Callable, List


class VideoFlip(ModalityTransform):
    """
    强制对视频进行水平翻转 (100% 触发)。
    用于配合已经离线翻转过 State/Action 的 Parquet 文件。
    """
    def __init__(self, apply_to):
        super().__init__(apply_to=apply_to)

    def apply(self, sample: dict) -> dict:
        """
        ModalityTransform 的 apply 方法接收整个 sample 字典。
        我们需要手动遍历 self.apply_to 并修改对应的 Tensor。
        """
        for key in self.apply_to:
            if key in sample:
                # 1. 获取 Tensor
                element = sample[key]
                
                # 2. 类型安全检查 (方便 Debug)
                if not isinstance(element, torch.Tensor):
                    # 如果 VideoToTensor 还没运行，这里可能是 numpy array 或 list
                    # 打印个警告，防止静默错误
                    print(f"[VideoFlip Warning] Key '{key}' is {type(element)}, expected Tensor.")
                    continue

                # 3. 执行翻转 (dims=[-1] 翻转宽度维度)
                sample[key] = torch.flip(element, dims=[-1])
                
        return sample


class VideoTransform(ModalityTransform):
    # Configurable attributes
    backend: str = Field(
        default="torchvision", description="The backend to use for the transformations"
    )

    # Model variables
    _train_transform: Callable | None = PrivateAttr(default=None)
    _eval_transform: Callable | None = PrivateAttr(default=None)
    _original_resolutions: dict[str, tuple[int, int]] = PrivateAttr(default_factory=dict)

    # Model constants
    _INTERPOLATION_MAP: ClassVar[dict[str, dict[str, Any]]] = PrivateAttr(
        {
            "nearest": {
                "albumentations": cv2.INTER_NEAREST,
                "torchvision": T.InterpolationMode.NEAREST,
            },
            "linear": {
                "albumentations": cv2.INTER_LINEAR,
                "torchvision": T.InterpolationMode.BILINEAR,
            },
            "cubic": {
                "albumentations": cv2.INTER_CUBIC,
                "torchvision": T.InterpolationMode.BICUBIC,
            },
            "area": {
                "albumentations": cv2.INTER_AREA,
                "torchvision": None,  # Torchvision does not support this interpolation mode
            },
            "lanczos4": {
                "albumentations": cv2.INTER_LANCZOS4,  # Lanczos with a 4x4 filter
                "torchvision": T.InterpolationMode.LANCZOS,  # Torchvision does not specify filter size, might be different from 4x4
            },
            "linear_exact": {
                "albumentations": cv2.INTER_LINEAR_EXACT,
                "torchvision": None,  # Torchvision does not support this interpolation mode
            },
            "nearest_exact": {
                "albumentations": cv2.INTER_NEAREST_EXACT,
                "torchvision": T.InterpolationMode.NEAREST_EXACT,
            },
            "max": {
                "albumentations": cv2.INTER_MAX,
                "torchvision": None,
            },
        }
    )

    @property
    def train_transform(self) -> Callable:
        assert (
            self._train_transform is not None
        ), "Transform is not set. Please call set_metadata() before calling apply()."
        return self._train_transform

    @train_transform.setter
    def train_transform(self, value: Callable):
        self._train_transform = value

    @property
    def eval_transform(self) -> Callable | None:
        return self._eval_transform

    @eval_transform.setter
    def eval_transform(self, value: Callable | None):
        self._eval_transform = value

    @property
    def original_resolutions(self) -> dict[str, tuple[int, int]]:
        assert (
            self._original_resolutions is not None
        ), "Original resolutions are not set. Please call set_metadata() before calling apply()."
        return self._original_resolutions

    @original_resolutions.setter
    def original_resolutions(self, value: dict[str, tuple[int, int]]):
        self._original_resolutions = value

    def check_input(self, data: dict[str, Any]):
        if self.backend == "torchvision":
            for key in self.apply_to:
                assert isinstance(data[key], torch.Tensor), f"Video {key} is not a torch tensor"
                assert data[key].ndim in [
                    4,
                    5,
                ], f"Expected video {key} to have 4 or 5 dimensions (T, C, H, W or T, B, C, H, W), got {data[key].ndim}"
        elif self.backend == "albumentations":
            for key in self.apply_to:
                assert isinstance(data[key], np.ndarray), f"Video {key} is not a numpy array"
                assert data[key].ndim in [
                    4,
                    5,
                ], f"Expected video {key} to have 4 or 5 dimensions (T, C, H, W or T, B, C, H, W), got {data[key].ndim}"
        else:
            raise ValueError(f"Backend {self.backend} not supported")

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        super().set_metadata(dataset_metadata)
        self.original_resolutions = {}
        for key in self.apply_to:
            split_keys = key.split(".")
            assert len(split_keys) == 2, f"Invalid key: {key}. Expected format: modality.key"
            sub_key = split_keys[1]
            if sub_key in dataset_metadata.modalities.video:
                self.original_resolutions[key] = dataset_metadata.modalities.video[
                    sub_key
                ].resolution
            else:
                raise ValueError(
                    f"Video key {sub_key} not found in dataset metadata. Available keys: {dataset_metadata.modalities.video.keys()}"
                )
        train_transform = self.get_transform(mode="train")
        eval_transform = self.get_transform(mode="eval")
        if self.backend == "albumentations":
            self.train_transform = A.ReplayCompose(transforms=[train_transform])  # type: ignore
            if eval_transform is not None:
                self.eval_transform = A.ReplayCompose(transforms=[eval_transform])  # type: ignore
        else:
            assert train_transform is not None, "Train transform must be set"
            self.train_transform = train_transform
            self.eval_transform = eval_transform

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        if self.training:
            transform = self.train_transform
            mode = 'train'
        else:
            transform = self.eval_transform
            mode = 'eval'
            if transform is None:
                return data
        assert (
            transform is not None
        ), "Transform is not set. Please call set_metadata() before calling apply()."
        try:
            self.check_input(data)
        except AssertionError as e:
            raise ValueError(
                f"Input data does not match the expected format for {self.__class__.__name__}: {e}"
            ) from e

        # Concatenate views
        views = [data[key] for key in self.apply_to]
        num_views = len(views)
        is_batched = views[0].ndim == 5
        bs = views[0].shape[0] if is_batched else 1
        if isinstance(views[0], torch.Tensor):
            views = torch.cat(views, 0)
        elif isinstance(views[0], np.ndarray):
            views = np.concatenate(views, 0)
        else:
            raise ValueError(f"Unsupported view type: {type(views[0])}")
        if is_batched:
            views = rearrange(views, "(v b) t c h w -> (v b t) c h w", v=num_views, b=bs)
        # Apply the transform
        if self.backend == "torchvision":
            views = transform(views)
        elif self.backend == "albumentations":
            assert isinstance(transform, A.ReplayCompose), "Transform must be a ReplayCompose"
            first_frame = views[0]
            transformed = transform(image=first_frame)
            replay_data = transformed["replay"]
            transformed_first_frame = transformed["image"]

            if len(views) > 1:
                # Apply the same transformations to the rest of the frames
                transformed_frames = [
                    transform.replay(replay_data, image=frame)["image"] for frame in views[1:]
                ]
                # Add the first frame back
                transformed_frames = [transformed_first_frame] + transformed_frames
            else:
                # If there is only one frame, just make a list with one frame
                transformed_frames = [transformed_first_frame]

            # Delete the replay data to save memory
            del replay_data
            views = np.stack(transformed_frames, 0)

        else:
            raise ValueError(f"Backend {self.backend} not supported")
        # Split views
        if is_batched:
            views = rearrange(views, "(v b t) c h w -> v b t c h w", v=num_views, b=bs)
        else:
            views = rearrange(views, "(v t) c h w -> v t c h w", v=num_views)
        for key, view in zip(self.apply_to, views):
            data[key] = view
        return data

    @classmethod
    def _validate_interpolation(cls, interpolation: str):
        if interpolation not in cls._INTERPOLATION_MAP:
            raise ValueError(f"Interpolation mode {interpolation} not supported")

    def _get_interpolation(self, interpolation: str, backend: str = "torchvision"):
        """
        Get the interpolation mode for the given backend.

        Args:
            interpolation (str): The interpolation mode.
            backend (str): The backend to use.

        Returns:
            Any: The interpolation mode for the given backend.
        """
        return self._INTERPOLATION_MAP[interpolation][backend]

    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable | None:
        raise NotImplementedError(
            "set_transform is not implemented for VideoTransform. Please implement this function to set the transforms."
        )

class VideoOffsetCrop(VideoTransform):
    """
    对视频帧进行指定位置和尺寸的裁剪。
    Applies a crop to the video frames at a specified location and size.
    """
    top: int = Field(..., description="裁剪区域上边缘的像素位置。")
    left: int = Field(..., description="裁剪区域左边缘的像素位置。")
    height: int = Field(..., description="裁剪区域的高度。")
    width: int = Field(..., description="裁剪区域的宽度。")

    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable:
        """
        获取裁剪变换函数。
        
        Args:
            mode (Literal["train", "eval"]): 模式（在此类中不影响行为）。

        Returns:
            Callable: 应用于视频张量的变换函数。
        """
        if self.backend == "torchvision":
            # torchvision.transforms.functional.crop 可以直接处理 (..., H, W) 的张量，
            # 因此它可以直接应用于 (T, C, H, W) 的视频张量，对每一帧进行相同的裁剪。
            def crop_fn(video_tensor: torch.Tensor) -> torch.Tensor:
                return F.crop(video_tensor, self.top, self.left, self.height, self.width)
            
            return Lambda(crop_fn)
        
        # 您可以根据需要为其他后端（如albumentations）添加实现
        elif self.backend == "albumentations":
            # Albumentations 的 Crop 需要不同的实现方式，这里暂时省略
            raise NotImplementedError("VideoOffsetCrop not implemented for albumentations backend yet.")
        else:
            raise ValueError(f"Backend {self.backend} not supported")

    def check_input(self, data: dict[str, Any]):
        # 您可以保留或根据需要实现输入检查
        pass

class VideoOmniUndistort(VideoTransform):
    d_list: List[float] = Field(..., description="Distortion parameters [p1, p2, k1, k2, k3, xi, fx, fy]")
    cx: float = Field(..., description="Principal point x")
    cy: float = Field(..., description="Principal point y")
    scale: float = Field(0.5, description="Scale factor for the new focal length (FOV control)")
    
    # 缓存生成的 grid，避免重复计算
    _grid: torch.Tensor | None = None

    def build_grid(self, h, w):
        """
        使用 Numpy 生成映射表，并转换为 PyTorch grid_sample 需要的格式 (H, W, 2) 范围 [-1, 1]
        """
        p1, p2, k1, k2, k3, xi, fx, fy = self.d_list
        
        # --- 你的原始核心算法 (复用) ---
        f_new = fx * self.scale
        K_new = np.array([
            [f_new, 0,     w/2],
            [0,     f_new, h/2],
            [0,     0,     1  ]
        ])
        
        grid_y, grid_x = np.mgrid[0:h, 0:w]
        grid_x = grid_x.astype(np.float32)
        grid_y = grid_y.astype(np.float32)

        # 步骤 A: 像素 -> 归一化
        x_norm = (grid_x - K_new[0, 2]) / K_new[0, 0]
        y_norm = (grid_y - K_new[1, 2]) / K_new[1, 1]
        
        # 步骤 B: 投影到单位球
        norm = np.sqrt(x_norm**2 + y_norm**2 + 1)
        Xs = x_norm / norm
        Ys = y_norm / norm
        Zs = 1.0 / norm 

        # 步骤 C: 坐标系变换 (xi)
        Z_prime = Zs + xi
        
        # 步骤 D: 再次投影到归一化平面
        x_prime = Xs / Z_prime
        y_prime = Ys / Z_prime
        
        # 步骤 E: 应用畸变
        r2 = x_prime**2 + y_prime**2
        r4 = r2**2
        r6 = r2**3
        
        radial_coeff = 1.0 + k1*r2 + k2*r4 + k3*r6
        dx = 2*p1*x_prime*y_prime + p2*(r2 + 2*x_prime**2)
        dy = p1*(r2 + 2*y_prime**2) + 2*p2*x_prime*y_prime
        
        x_distorted = x_prime * radial_coeff + dx
        y_distorted = y_prime * radial_coeff + dy
        
        # 步骤 F: 映射回源像素
        map_x = x_distorted * fx + self.cx
        map_y = y_distorted * fy + self.cy
        
        # --- 转换为 PyTorch grid_sample 格式 ---
        # grid_sample 要求坐标范围在 [-1, 1] 之间
        # x: -1 (左) -> 1 (右)
        # y: -1 (上) -> 1 (下)
        grid_x_norm = 2.0 * map_x / (w - 1) - 1.0
        grid_y_norm = 2.0 * map_y / (h - 1) - 1.0
        
        # 堆叠并转为 Tensor (H, W, 2)
        grid = np.stack((grid_x_norm, grid_y_norm), axis=-1).astype(np.float32)
        return torch.from_numpy(grid)

    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable:
        # 1. 获取原始分辨率
        assert len(set(self.original_resolutions.values())) == 1, "All video keys must have same resolution"
        # 这里的 apply_to[0] 对应你的 video key
        w_orig, h_orig = self.original_resolutions[self.apply_to[0]]
        
        # 2. 如果 grid 还没创建，创建一次
        if self._grid is None:
            self._grid = self.build_grid(h_orig, w_orig)
        
        # 3. 定义实际执行的函数
        def apply_undistort(video_tensor: torch.Tensor):
            """
            Args:
                video_tensor: (T, C, H, W) 或者是 (C, H, W)
            Returns:
                Undistorted tensor
            """
            # 确保 grid 在同一个 device 上
            grid = self._grid.to(video_tensor.device)
            
            # 处理输入维度
            is_batched = video_tensor.ndim == 4
            if not is_batched:
                # (C, H, W) -> (1, C, H, W)
                video_tensor = video_tensor.unsqueeze(0)
            
            # Expand grid to match batch size: (T, H, W, 2)
            T = video_tensor.shape[0]
            current_grid = grid.unsqueeze(0).expand(T, -1, -1, -1)
            
            # 执行 Grid Sample (等价于 cv2.remap)
            # align_corners=True 通常更精确
            out = torch.nn.functional.grid_sample(video_tensor, current_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
            
            if not is_batched:
                out = out.squeeze(0)
                
            return out

        return apply_undistort


class VideoCrop(VideoTransform):
    height: int | None = Field(default=None, description="The height of the input image")
    width: int | None = Field(default=None, description="The width of the input image")
    scale: float = Field(
        ...,
        description="The scale of the crop. The crop size is (width * scale, height * scale)",
    )

    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable:
        """Get the transform for the given mode.

        Args:
            mode (Literal["train", "eval"]): The mode to get the transform for.

        Returns:
            Callable: If mode is "train", return a random crop transform. If mode is "eval", return a center crop transform.
        """
        # 1. Check the input resolution
        assert (
            len(set(self.original_resolutions.values())) == 1
        ), f"All video keys must have the same resolution, got: {self.original_resolutions}"
        if self.height is None:
            assert self.width is None, "Height and width must be either both provided or both None"
            self.width, self.height = self.original_resolutions[self.apply_to[0]]
        else:
            assert (
                self.width is not None
            ), "Height and width must be either both provided or both None"
        # 2. Create the transform
        size = (int(self.height * self.scale), int(self.width * self.scale))
        if self.backend == "torchvision":
            if mode == "train":
                return T.RandomCrop(size)
            elif mode == "eval":
                return T.CenterCrop(size)
            else:
                raise ValueError(f"Crop mode {mode} not supported")
        elif self.backend == "albumentations":
            if mode == "train":
                return A.RandomCrop(height=size[0], width=size[1], p=1)
            elif mode == "eval":
                return A.CenterCrop(height=size[0], width=size[1], p=1)
            else:
                raise ValueError(f"Crop mode {mode} not supported")
        else:
            raise ValueError(f"Backend {self.backend} not supported")

    def check_input(self, data: dict[str, Any]):
        super().check_input(data)
        # Check the input resolution
        for key in self.apply_to:
            if self.backend == "torchvision":
                height, width = data[key].shape[-2:]
            elif self.backend == "albumentations":
                height, width = data[key].shape[-3:-1]
            else:
                raise ValueError(f"Backend {self.backend} not supported")
            if self.training:
                assert (
                    height == self.height and width == self.width
                ), f"Video {key} has invalid shape {height, width}, expected {self.height, self.width}"


class VideoResizeCropRec(VideoTransform):
    height: int = Field(..., description="The target size (height) of the crop")
    width: int = Field(..., description="The target size (width) of the crop")
    interpolation: str = Field(default="linear", description="The interpolation mode")
    antialias: bool = Field(default=True, description="Whether to apply antialiasing")

    @field_validator("interpolation")
    def validate_interpolation(cls, v):
        cls._validate_interpolation(v)
        return v

    def _resize_by_height_torch(self, img) -> Any:
        """
        Helper function for Torchvision: 
        Resize image to fixed self.height while maintaining aspect ratio.
        """
        # 获取插值模式 (通常 _get_interpolation 返回的是 Enum，如 InterpolationMode.BILINEAR)
        interp = self._get_interpolation(self.interpolation, "torchvision")
        
        # 获取原图尺寸 (W, H)
        w, h = F.get_image_size(img)
        
        # 如果高度已经符合，直接返回（节省计算）
        if h == self.height:
            return img
            
        # 计算新的宽度：New Width = Old Width * (Target Height / Old Height)
        new_w = int(w * (self.height / h))
        
        # 执行 Resize，强制指定 [height, width]
        return F.resize(
            img, 
            [self.height, new_w], 
            interpolation=interp, 
            antialias=self.antialias
        )

    def _resize_by_height_alb(self, image, **kwargs) -> Any:
        """
        Helper function for Albumentations:
        Resize image to fixed self.height while maintaining aspect ratio.
        """
        # 获取插值模式 (通常是 cv2.INTER_LINEAR 等 int 值)
        interp = self._get_interpolation(self.interpolation, "albumentations")
        
        h, w = image.shape[:2]
        
        if h == self.height:
            return image
            
        scale = self.height / h
        new_w = int(w * scale)
        
        # cv2.resize 参数顺序是 (width, height)
        return cv2.resize(image, (new_w, self.height), interpolation=interp)

    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable:
        """Get the transform: Fixed Height Resize -> Center Crop Width."""
        
        # 预先检查插值是否合法
        if self._get_interpolation(self.interpolation, self.backend) is None:
             raise ValueError(f"Interpolation {self.interpolation} not supported")

        if self.backend == "torchvision":
            return T.Compose([
                # 1. 调用上面封装好的 Torchvision 缩放函数
                T.Lambda(self._resize_by_height_torch),
                # 2. 中心裁剪 (height, width)
                T.CenterCrop((self.height, self.width))
            ])
            
        elif self.backend == "albumentations":
            return A.Compose([
                # 1. 调用上面封装好的 Albumentations 缩放函数
                A.Lambda(name="resize_height_fixed", image=self._resize_by_height_alb),
                # 2. 中心裁剪
                A.CenterCrop(height=self.height, width=self.width, p=1)
            ])
            
        else:
            raise ValueError(f"Backend {self.backend} not supported")


class VideoResizeCropSquare(VideoTransform):
    # 如果目标是正方形 (size, size)，通常 height 和 width 设为相同的值
    height: int = Field(..., description="The target size (height) of the crop")
    width: int = Field(..., description="The target size (width) of the crop")
    interpolation: str = Field(default="linear", description="The interpolation mode")
    antialias: bool = Field(default=True, description="Whether to apply antialiasing")

    @field_validator("interpolation")
    def validate_interpolation(cls, v):
        cls._validate_interpolation(v)
        return v

    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable:
        """Get the resize + center crop transform.
        
        Logic:
        1. Resize the shorter edge to 'size'.
        2. Center crop a square of 'size x size'.
        """
        interpolation = self._get_interpolation(self.interpolation, self.backend)
        if interpolation is None:
            raise ValueError(
                f"Interpolation mode {self.interpolation} not supported for torchvision"
            )
            
        # 假设输出是正方形，取 height 作为基准 size
        size = self.height 

        if self.backend == "torchvision":
            # torchvision 的 T.Resize 如果接收一个 int，
            # 它会自动将图像的短边 resize 到这个 int，并保持长宽比
            return T.Compose([
                T.Resize(size, interpolation=interpolation, antialias=self.antialias),
                T.CenterCrop(size)  # 裁剪出 (size, size)
            ])
            
        elif self.backend == "albumentations":
            # albumentations 需要显式使用 SmallestMaxSize 来调整短边
            return A.Compose([
                A.SmallestMaxSize(max_size=size, interpolation=interpolation, p=1),
                A.CenterCrop(height=size, width=size, p=1)
            ])
            
        else:
            raise ValueError(f"Backend {self.backend} not supported")


class VideoResize(VideoTransform):
    height: int = Field(..., description="The height of the resize")
    width: int = Field(..., description="The width of the resize")
    interpolation: str = Field(default="linear", description="The interpolation mode")
    antialias: bool = Field(default=True, description="Whether to apply antialiasing")

    @field_validator("interpolation")
    def validate_interpolation(cls, v):
        cls._validate_interpolation(v)
        return v

    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable:
        """Get the resize transform. Same transform for both train and eval.

        Args:
            mode (Literal["train", "eval"]): The mode to get the transform for.

        Returns:
            Callable: The resize transform.
        """
        interpolation = self._get_interpolation(self.interpolation, self.backend)
        if interpolation is None:
            raise ValueError(
                f"Interpolation mode {self.interpolation} not supported for torchvision"
            )
        if self.backend == "torchvision":
            size = (self.height, self.width)
            return T.Resize(size, interpolation=interpolation, antialias=self.antialias)
        elif self.backend == "albumentations":
            return A.Resize(
                height=self.height,
                width=self.width,
                interpolation=interpolation,
                p=1,
            )
        else:
            raise ValueError(f"Backend {self.backend} not supported")


class VideoRandomRotation(VideoTransform):
    degrees: float | tuple[float, float] = Field(
        ..., description="The degrees of the random rotation"
    )
    interpolation: str = Field("linear", description="The interpolation mode")

    @field_validator("interpolation")
    def validate_interpolation(cls, v):
        cls._validate_interpolation(v)
        return v

    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable | None:
        """Get the random rotation transform, only used in train mode.

        Args:
            mode (Literal["train", "eval"]): The mode to get the transform for.

        Returns:
            Callable | None: The random rotation transform. None for eval mode.
        """
        if mode == "eval":
            return None
        interpolation = self._get_interpolation(self.interpolation, self.backend)
        if interpolation is None:
            raise ValueError(
                f"Interpolation mode {self.interpolation} not supported for torchvision"
            )
        if self.backend == "torchvision":
            return T.RandomRotation(self.degrees, interpolation=interpolation)  # type: ignore
        elif self.backend == "albumentations":
            return A.Rotate(limit=self.degrees, interpolation=interpolation, p=1)
        else:
            raise ValueError(f"Backend {self.backend} not supported")


class VideoHorizontalFlip(VideoTransform):
    p: float = Field(..., description="The probability of the horizontal flip")

    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable | None:
        """Get the horizontal flip transform, only used in train mode.

        Args:
            mode (Literal["train", "eval"]): The mode to get the transform for.

        Returns:
            Callable | None: If mode is "train", return a horizontal flip transform. If mode is "eval", return None.
        """
        if mode == "eval":
            return None
        if self.backend == "torchvision":
            return T.RandomHorizontalFlip(self.p)
        elif self.backend == "albumentations":
            return A.HorizontalFlip(p=self.p)
        else:
            raise ValueError(f"Backend {self.backend} not supported")


class VideoGrayscale(VideoTransform):
    p: float = Field(..., description="The probability of the grayscale transformation")

    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable | None:
        """Get the grayscale transform, only used in train mode.

        Args:
            mode (Literal["train", "eval"]): The mode to get the transform for.

        Returns:
            Callable | None: If mode is "train", return a grayscale transform. If mode is "eval", return None.
        """
        if mode == "eval":
            return None
        if self.backend == "torchvision":
            return T.RandomGrayscale(self.p)
        elif self.backend == "albumentations":
            return A.ToGray(p=self.p)
        else:
            raise ValueError(f"Backend {self.backend} not supported")


class VideoColorJitter(VideoTransform):
    brightness: float | tuple[float, float] = Field(
        ..., description="The brightness of the color jitter"
    )
    contrast: float | tuple[float, float] = Field(
        ..., description="The contrast of the color jitter"
    )
    saturation: float | tuple[float, float] = Field(
        ..., description="The saturation of the color jitter"
    )
    hue: float | tuple[float, float] = Field(..., description="The hue of the color jitter")

    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable | None:
        """Get the color jitter transform, only used in train mode.

        Args:
            mode (Literal["train", "eval"]): The mode to get the transform for.

        Returns:
            Callable | None: If mode is "train", return a color jitter transform. If mode is "eval", return None.
        """
        if mode == "eval":
            return None
        if self.backend == "torchvision":
            return T.ColorJitter(
                brightness=self.brightness,
                contrast=self.contrast,
                saturation=self.saturation,
                hue=self.hue,
            )
        elif self.backend == "albumentations":
            return A.ColorJitter(
                brightness=self.brightness,
                contrast=self.contrast,
                saturation=self.saturation,
                hue=self.hue,
                p=1,
            )
        else:
            raise ValueError(f"Backend {self.backend} not supported")


class VideoRandomGrayscale(VideoTransform):
    p: float = Field(..., description="The probability of the grayscale transformation")

    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable | None:
        """Get the grayscale transform, only used in train mode.

        Args:
            mode (Literal["train", "eval"]): The mode to get the transform for.

        Returns:
            Callable | None: If mode is "train", return a grayscale transform. If mode is "eval", return None.
        """
        if mode == "eval":
            return None
        if self.backend == "torchvision":
            return T.RandomGrayscale(self.p)
        elif self.backend == "albumentations":
            return A.ToGray(p=self.p)
        else:
            raise ValueError(f"Backend {self.backend} not supported")


class VideoRandomPosterize(VideoTransform):
    bits: int = Field(..., description="The number of bits to posterize the image")
    p: float = Field(..., description="The probability of the posterize transformation")

    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable | None:
        """Get the posterize transform, only used in train mode.

        Args:
            mode (Literal["train", "eval"]): The mode to get the transform for.

        Returns:
            Callable | None: If mode is "train", return a posterize transform. If mode is "eval", return None.
        """
        if mode == "eval":
            return None
        if self.backend == "torchvision":
            return T.RandomPosterize(bits=self.bits, p=self.p)
        elif self.backend == "albumentations":
            return A.Posterize(num_bits=self.bits, p=self.p)
        else:
            raise ValueError(f"Backend {self.backend} not supported")


class VideoToTensor(VideoTransform):
    check_resolution: bool = Field(
        default=True,
        description="If False, skip metadata resolution validation before converting to tensor.",
    )

    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable:
        """Get the to tensor transform. Same transform for both train and eval.

        Args:
            mode (Literal["train", "eval"]): The mode to get the transform for.

        Returns:
            Callable: The to tensor transform.
        """
        if self.backend == "torchvision":
            return self.__class__.to_tensor
        else:
            raise ValueError(f"Backend {self.backend} not supported")

    def check_input(self, data: dict):
        """Check if the input data has the correct shape.
        Expected video shape: [T, H, W, C], dtype np.uint8
        """
        for key in self.apply_to:
            assert key in data, f"Key {key} not found in data. Available keys: {data.keys()}"
            assert data[key].ndim in [
                4,
                5,
            ], f"Video {key} must have 4 or 5 dimensions, got {data[key].ndim}"
            assert (
                data[key].dtype == np.uint8
            ), f"Video {key} must have dtype uint8, got {data[key].dtype}"

            if self.training and self.check_resolution:
                input_resolution = data[key].shape[-3:-1][::-1]
                if key in self.original_resolutions:
                    expected_resolution = self.original_resolutions[key]
                else:
                    expected_resolution = input_resolution
                assert (
                    input_resolution == expected_resolution
                ), f"Video {key} has invalid resolution {input_resolution}, expected {expected_resolution}. Full shape: {data[key].shape}"

    @staticmethod
    def to_tensor(frames: np.ndarray) -> torch.Tensor:
        """Convert numpy array to tensor efficiently.

        Args:
            frames: numpy array of shape [T, H, W, C] in uint8 format
        Returns:
            tensor of shape [T, C, H, W] in range [0, 1]
        """
        frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).to(torch.float32)
        frames_tensor.div_(255.0)
        return frames_tensor  # [T, C, H, W]


class VideoToNumpy(VideoTransform):
    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable:
        """Get the to numpy transform. Same transform for both train and eval.

        Args:
            mode (Literal["train", "eval"]): The mode to get the transform for.

        Returns:
            Callable: The to numpy transform.
        """
        if self.backend == "torchvision":
            return self.__class__.to_numpy
        else:
            raise ValueError(f"Backend {self.backend} not supported")

    @staticmethod
    def to_numpy(frames: torch.Tensor) -> np.ndarray:
        """Convert tensor back to numpy array efficiently.

        Args:
            frames: tensor of shape [T, C, H, W] in range [0, 1]
        Returns:
            numpy array of shape [T, H, W, C] in uint8 format
        """
        return (frames.permute(0, 2, 3, 1) * 255).to(torch.uint8).cpu().numpy().copy()

class VideoToPIL(VideoTransform):
    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable:
        """Get the to PIL transform. Same transform for both train and eval.

        Args:
            mode (Literal["train", "eval"]): The mode to get the transform for.

        Returns:
            Callable: The to PIL transform.
        """
        if self.backend == "torchvision":
            return self.__class__.to_pil
        else:
            raise ValueError(f"Backend {self.backend} not supported")

    @staticmethod
    def to_pil(frames: torch.Tensor) -> Image.Image:
        """Convert tensor back to PIL Image.

        Args:
            frames: tensor of shape [T, C, H, W] in range [0, 1]
        Returns:
            PIL Image of shape [T, H, W, C] in uint8 format
        """
        # video PIL format?
        return Image.fromarray((frames.permute(0, 2, 3, 1) * 255).to(torch.uint8).cpu().numpy())