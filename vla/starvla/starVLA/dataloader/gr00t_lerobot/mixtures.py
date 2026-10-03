"""
mixtures.py

Registry of dataset mixtures: name -> [(dataset_subdir, sampling_weight, robot_type), ...].
``dataset_subdir`` is relative to ``datasets.vla_data.data_root_dir`` (the LeRobot root of
nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim); ``robot_type`` indexes ROBOT_TYPE_CONFIG_MAP.
"""

from typing import Dict, List, Tuple

DATASET_NAMED_MIXTURES: Dict[str, List[Tuple[str, float, str]]] = {
    "gr1_multi_concat": [
        ("gr1_unified.PnPPotatoToMicrowaveClose", 1.0, "fourier_gr1_arms_waist_concat_starvla"),
        ("gr1_unified.PnPMilkToMicrowaveClose", 1.0, "fourier_gr1_arms_waist_concat_starvla"),
        ("gr1_unified.PnPCanToDrawerClose", 1.0, "fourier_gr1_arms_waist_concat_starvla"),
        ("gr1_unified.PnPCupToDrawerClose", 1.0, "fourier_gr1_arms_waist_concat_starvla"),
        ("gr1_unified.PnPBottleToCabinetClose", 1.0, "fourier_gr1_arms_waist_concat_starvla"),
        ("gr1_unified.PnPWineToCabinetClose", 1.0, "fourier_gr1_arms_waist_concat_starvla"),
        ("gr1_unified.PosttrainPnPNovelFromPlacematToBowlSplitA", 1.0, "fourier_gr1_arms_waist_concat_starvla"),
        ("gr1_unified.PosttrainPnPNovelFromPlateToPlateSplitA", 1.0, "fourier_gr1_arms_waist_concat_starvla"),
        ("gr1_unified.PosttrainPnPNovelFromPlacematToPlateSplitA", 1.0, "fourier_gr1_arms_waist_concat_starvla"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToPotSplitA", 1.0, "fourier_gr1_arms_waist_concat_starvla"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA", 1.0, "fourier_gr1_arms_waist_concat_starvla"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToPanSplitA", 1.0, "fourier_gr1_arms_waist_concat_starvla"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToCardboardboxSplitA", 1.0, "fourier_gr1_arms_waist_concat_starvla"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToTieredshelfSplitA", 1.0, "fourier_gr1_arms_waist_concat_starvla"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA", 1.0, "fourier_gr1_arms_waist_concat_starvla"),
        ("gr1_unified.PosttrainPnPNovelFromPlacematToTieredshelfSplitA", 1.0, "fourier_gr1_arms_waist_concat_starvla"),
        ("gr1_unified.PosttrainPnPNovelFromPlateToCardboardboxSplitA", 1.0, "fourier_gr1_arms_waist_concat_starvla"),
        ("gr1_unified.PosttrainPnPNovelFromPlacematToBasketSplitA", 1.0, "fourier_gr1_arms_waist_concat_starvla"),
        ("gr1_unified.PosttrainPnPNovelFromPlateToPanSplitA", 1.0, "fourier_gr1_arms_waist_concat_starvla"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToTieredbasketSplitA", 1.0, "fourier_gr1_arms_waist_concat_starvla"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToPotSplitA", 1.0, "fourier_gr1_arms_waist_concat_starvla"),
        ("gr1_unified.PosttrainPnPNovelFromPlateToBowlSplitA", 1.0, "fourier_gr1_arms_waist_concat_starvla"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToBasketSplitA", 1.0, "fourier_gr1_arms_waist_concat_starvla"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToPlateSplitA", 1.0, "fourier_gr1_arms_waist_concat_starvla"),
    ],
}
