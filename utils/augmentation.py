# SRPU-Model/utils/augmentation.py

import json
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from typing import Optional, List, Tuple, Dict, Any, Union
import sys

# --- 配置导入 ---
try:
    from configs import base as cfg_base
    from configs import json_config as cfg_json_gen
    from configs import augmentation_config as cfg_aug # <-- 导入新的增强配置
except ImportError:
    if __name__ == '__main__' and __package__ is None:
        current_script_path = Path(__file__).resolve()
        project_root_for_config = current_script_path.parent.parent
        if str(project_root_for_config) not in sys.path:
            sys.path.insert(0, str(project_root_for_config))
        from configs import base as cfg_base
        from configs import json_config as cfg_json_gen
        from configs import augmentation_config as cfg_aug # <-- 导入新的增强配置
    else:
        raise

def load_dataset_stats(
    stats_json_name: str = "dataset_stats.json",
    project_root: Optional[Path] = None,
    json_dir_name_relative_to_project: Optional[str] = None,
    expected_input_depth: Optional[int] = None
) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """从 dataset_stats.json 文件加载均值和标准差。(此函数无变化)"""
    # ... (此函数代码与上一版本相同，此处省略)
    if project_root is None:
        project_root = cfg_base.PROJECT_ROOT.resolve()
    if json_dir_name_relative_to_project is None:
        json_dir_name_relative_to_project = cfg_json_gen.JSON_OUTPUT_DIR_NAME
    stats_file_path = project_root / json_dir_name_relative_to_project / stats_json_name
    if not stats_file_path.exists():
        print(f"警告: 统计文件未找到: {stats_file_path}")
        return None, None
    try:
        with open(stats_file_path, 'r') as f:
            stats_data = json.load(f)
        mean, std = stats_data.get("mean"), stats_data.get("std")
        input_depth_at_calc = stats_data.get("input_depth_at_calculation")
        if expected_input_depth is None:
            expected_input_depth = cfg_base.INPUT_DEPTH
        if input_depth_at_calc != expected_input_depth or len(mean) != expected_input_depth:
             print(f"警告: 统计数据维度 ({len(mean)}) 与当前配置 ({expected_input_depth}) 不匹配。将不使用归一化。")
             return None, None
        return mean, std
    except Exception as e:
        print(f"加载统计文件 {stats_file_path} 时出错: {e}")
        return None, None


def _get_aug(aug_name: str, params: Dict[str, Any]) -> Optional[A.BasicTransform]:
    """根据名称和参数返回一个albumentations变换实例。"""
    if not params.get("enabled", False):
        return None
    
    # 从参数字典中移除 'enabled'，剩下的传递给变换函数
    params.pop("enabled", None)

    # 映射表，将配置名映射到albumentations类
    AUGMENTATION_MAP = {
        "random_crop": A.RandomCrop,
        "center_crop": A.CenterCrop,
        "horizontal_flip": A.HorizontalFlip,
        "vertical_flip": A.VerticalFlip,
        "random_rotate_90": A.RandomRotate90,
        "rotate": A.Rotate,
        "random_brightness_contrast": A.RandomBrightnessContrast,
        "gaussian_blur": A.GaussianBlur,
        "gauss_noise": A.GaussNoise,
    }
    
    aug_class = AUGMENTATION_MAP.get(aug_name)
    if aug_class:
        return aug_class(**params)
    return None

def build_augmentations(
    mode: str, # "train", "val", or "ssl"
    height: int, 
    width: int,
    use_normalization: bool = True # SSL时可以关闭归一化
) -> A.Compose:
    """
    根据配置文件动态构建训练、验证或SSL的增强流水线。
    """
    if mode == "train":
        config = cfg_aug.TRAIN_AUGMENTATIONS
    elif mode == "val":
        config = cfg_aug.VAL_AUGMENTATIONS
    elif mode == "ssl":
        config = cfg_aug.SSL_AUGMENTATIONS
    else:
        raise ValueError(f"未知的增强模式: {mode}")

    # 动态添加裁剪尺寸
    if "random_crop" in config:
        config["random_crop"]["height"] = height
        config["random_crop"]["width"] = width
    if "center_crop" in config:
        config["center_crop"]["height"] = height
        config["center_crop"]["width"] = width
    
    # 根据配置构建变换列表
    transforms_list = []
    for aug_name, params in config.items():
        aug_instance = _get_aug(aug_name, params.copy()) # 传入副本以防修改原始配置
        if aug_instance:
            transforms_list.append(aug_instance)
            
    # 添加归一化和Tensor转换
    if use_normalization:
        mean, std = load_dataset_stats(expected_input_depth=cfg_base.INPUT_DEPTH)
        if mean and std:
            transforms_list.append(A.Normalize(mean=mean, std=std, max_pixel_value=1.0))

    transforms_list.append(ToTensorV2())
    
    print(f"--- 为模式 '{mode}' 构建的增强流水线 ---")
    for t in transforms_list:
        print(f"  - {t.__class__.__name__}")
    print("------------------------------------------")
    
    return A.Compose(transforms_list)