from pathlib import Path
from typing import Sequence
from omegaconf import OmegaConf

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset, LeRobotMixtureDataset
from starVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
from starVLA.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import ROBOT_TYPE_TO_EMBODIMENT_TAG, EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.norm_scheme import (
    DEFAULT_ACTION_NORM_SCHEME,
    apply_norm_scheme_to_mixture,
)

def collate_fn(batch):
    return batch

def make_LeRobotSingleDataset(
    data_root_dir: Path | str,
    data_name: str,
    robot_type: str,  # 新增参数
    delete_pause_frame: bool = False,
    num_shot: int = None, #每个dataset是否只读前num_shot条
    data_percent: float | None = None,
    subset_seed: int = 42,
    use_delta: bool = False, #是否使用delta的action，这里代码有问题，不要直接使用
    use_simple_tasks: bool = False, #是否使用tasks_simple.jsonl进行任务增强
    video_max_height: int = None, #decord读取视频时直接resize的最大高度
    video_max_width: int = None, #decord读取视频时直接resize的最大宽度
    stats_cache_dir: Path | str | None = None,
) -> LeRobotSingleDataset:
    """
    Make a LeRobotSingleDataset object.

    :param data_root_dir: The root directory of the dataset.
    :param data_name: The name of the dataset.
    :param robot_type: The robot type config to use.
    :param crop_obs_camera: Whether to crop the observation camera images.
    :param use_simple_tasks: Whether to use tasks_simple.jsonl for task augmentation.
    :param video_max_height: Max height for decord to resize video during decode. None means no resize.
    :param video_max_width: Max width for decord to resize video during decode. None means no resize.
    :return: A LeRobotSingleDataset object.
    """
    
    data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
    print("-------------------------")
    print("data_config:", data_config)
    print("-------------------------")
    
    modality_config = data_config.modality_config()
    transforms = data_config.transform()
    dataset_path = data_root_dir / data_name
    target_fps = getattr(data_config, "target_fps", None)
    info_filename = getattr(data_config, "info_filename", None)
    modality_filename = getattr(data_config, "modality_filename", None)
    use_episode_instruction = getattr(data_config, "use_episode_instruction", True)
    strip_bilingual_task_prefix = getattr(data_config, "strip_bilingual_task_prefix", False)

    if robot_type not in ROBOT_TYPE_TO_EMBODIMENT_TAG:
        print(f"Warning: Robot type {robot_type} not found in ROBOT_TYPE_TO_EMBODIMENT_TAG, using {EmbodimentTag.NEW_EMBODIMENT} as default")
        embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    else:
        embodiment_tag = ROBOT_TYPE_TO_EMBODIMENT_TAG[robot_type]
    
    # Build video_backend_kwargs for decord resize during decode
    video_backend_kwargs = {"num_threads": 4}
    decode_size = getattr(data_config, "decode_size", None)
    if decode_size is not None:
        video_backend_kwargs["height"], video_backend_kwargs["width"] = decode_size[0], decode_size[1]
    if video_max_height is not None:
        video_backend_kwargs["height"] = video_max_height
    if video_max_width is not None:
        video_backend_kwargs["width"] = video_max_width
    
    return LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend="decord",
        video_backend_kwargs=video_backend_kwargs,
        delete_pause_frame=delete_pause_frame,
        num_shot=num_shot,
        data_percent=data_percent,
        subset_seed=subset_seed,
        use_delta=use_delta,
        use_simple_tasks=use_simple_tasks,
        target_fps=target_fps,
        info_filename=info_filename,
        modality_filename=modality_filename,
        use_episode_instruction=use_episode_instruction,
        strip_bilingual_task_prefix=strip_bilingual_task_prefix,
        stats_cache_dir=stats_cache_dir,
    )

def get_vla_dataset(
    data_cfg: dict,
    mode: str = "train",
    balance_dataset_weights: bool = True,
    balance_trajectory_weights: bool = True,
    metadata_config: dict = {
        "percentile_mixing_method": "weighted_average",
    },
    seed: int = 42,
    delete_pause_frame: bool = False,
    **kwargs: dict,
) -> LeRobotMixtureDataset:
    """
    Get a LeRobotMixtureDataset object.
    """
    data_root_dir = data_cfg.data_root_dir
    data_mix = data_cfg.data_mix
    image_size = tuple(data_cfg.image_size)
    num_shot = data_cfg.get("num_shot", None) #每个dataset是否只读前num_shot条
    data_percent = data_cfg.get("data_percent", None)
    subset_seed = data_cfg.get("subset_seed", 42)
    use_delta = data_cfg.get("use_delta", False)#是否使用delta的action，这里代码有问题，不要直接使用
    dynamic_image_size = data_cfg.get("dynamic_image_size", False) #是否跳过强制resize，保持data_config输出的原始尺寸
    use_simple_tasks = data_cfg.get("use_simple_tasks", False) #是否使用tasks_simple.jsonl进行任务增强
    cached_statistics_path = data_cfg.get("cached_statistics_path", None) #复用预训练的统计量，用于finetune时避免从小数据集重新计算
    stats_cache_dir = data_cfg.get("stats_cache_dir", None) #可写的stats缓存目录，用于数据集目录为只读时
    video_max_height = data_cfg.get("video_max_height", None) #decord读取视频时直接resize的最大高度
    video_max_width = data_cfg.get("video_max_width", None) #decord读取视频时直接resize的最大宽度
    disable_separate_embodiment_norm = data_cfg.get("disable_separate_embodiment_norm", False) #禁用按embodiment分组的normalization，改为全局合并统计量
    # Runtime action norm scheme. Default mean_std keeps DataConfig defaults.
    # Set ori_nonorm_q99 to leave rot/ori6d unnormalized and use q99 elsewhere.
    action_norm_scheme = str(
        data_cfg.get("action_norm_scheme", DEFAULT_ACTION_NORM_SCHEME)
    ).lower().strip()
    max_action_dim = data_cfg.get("max_action_dim", None)
    max_state_dim = data_cfg.get("max_state_dim", None)

    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]
    included_datasets, filtered_mixture_spec = set(), []
    for d_name, d_weight, robot_type in mixture_spec:  
        dataset_key = (d_name, robot_type)  
        if dataset_key in included_datasets:
            print(f"Skipping Duplicate Dataset: `{(d_name, d_weight, robot_type)}`")
            continue

        included_datasets.add(dataset_key)
        filtered_mixture_spec.append((d_name, d_weight, robot_type))

    dataset_mixture = []
    for d_name, d_weight, robot_type in filtered_mixture_spec:
        dataset_mixture.append((make_LeRobotSingleDataset(Path(data_root_dir), d_name, robot_type, delete_pause_frame=delete_pause_frame, num_shot=num_shot, data_percent=data_percent, subset_seed=subset_seed, use_delta=use_delta, use_simple_tasks=use_simple_tasks, video_max_height=video_max_height, video_max_width=video_max_width, stats_cache_dir=stats_cache_dir), d_weight))

    mixture_kwargs = dict(
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        metadata_config=metadata_config,
        image_size=image_size,
        use_delta=use_delta,
        dynamic_image_size=dynamic_image_size,
        seed=seed,
        cached_statistics_path=cached_statistics_path,
        disable_separate_embodiment_norm=disable_separate_embodiment_norm,
        max_action_dim=max_action_dim,
        max_state_dim=max_state_dim,
        **kwargs,
    )

    mixture = LeRobotMixtureDataset(dataset_mixture, **mixture_kwargs)

    # Must run after mixture init (update_metadata / set_transforms_metadata).
    n_patched = apply_norm_scheme_to_mixture(mixture, action_norm_scheme)
    print(
        f"[get_vla_dataset] action_norm_scheme={action_norm_scheme} "
        f"patched_transforms={n_patched}"
    )
    return mixture
