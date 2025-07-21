# SRPU-Model/datasets/sem_datasets.py
# --- 修正并融合了新逻辑的最终版本 ---

import json
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable, Tuple, List
import sys
import cv2

# --- 导入 ---
try:
    from configs import base as cfg_base
    from configs import json_config as cfg_json
    from configs.dataset_config import SPLIT_RATIO, SPLIT_SEED
    from utils.augmentation import build_augmentations 
except ImportError:
    if __name__ == '__main__' and __package__ is None:
        current_script_path = Path(__file__).resolve()
        project_root_for_import = current_script_path.parent.parent
        if str(project_root_for_import) not in sys.path:
            sys.path.insert(0, str(project_root_for_import))
        from configs import base as cfg_base
        from configs import json_config as cfg_json
        from configs.dataset_config import SPLIT_RATIO, SPLIT_SEED
        from utils.augmentation import build_augmentations
    else:
        raise

class SemSegmentationDataset(Dataset):
    """
    用于SEM图像分割的数据集类 (重构优化版)。
    - 正确解析由 json_generator.py 生成的扁平化JSON。
    - 支持动态划分train/val/test。
    - 在 __getitem__ 中动态加载和堆叠3D图像。
    - 支持 albumentations 进行数据增强。
    """
    def __init__(self,
                 json_file_identifier: str,
                 project_root: Path,
                 split: str = 'train',
                 split_ratio: Optional[tuple] = None,
                 seed: Optional[int] = None,
                 return_dataset_name: bool = False): # 新增参数
        self.project_root = project_root
        self.split = split
        self.split_ratio = split_ratio if split_ratio is not None else SPLIT_RATIO
        self.seed = seed if seed is not None else SPLIT_SEED
        self.return_dataset_name = return_dataset_name # 保存参数

        # 构建完整的JSON文件路径
        self.json_path = self.project_root / cfg_json.JSON_OUTPUT_DIR_NAME / json_file_identifier
        if not self.json_path.exists():
            raise FileNotFoundError(f"指定的JSON文件未找到: {self.json_path}")

        # 加载JSON数据
        with open(self.json_path, 'r') as f:
            all_samples = json.load(f).get("samples", [])
        num_samples = len(all_samples)
        idxs = list(range(num_samples))
        import random
        random.seed(self.seed)
        random.shuffle(idxs)
        n_train = int(num_samples * self.split_ratio[0])
        n_val = int(num_samples * self.split_ratio[1])
        n_test = num_samples - n_train - n_val
        if self.split == 'train':
            selected = idxs[:n_train]
        elif self.split == 'val':
            selected = idxs[n_train:n_train+n_val]
        elif self.split == 'test':
            selected = idxs[n_train+n_val:]
        else:
            raise ValueError(f"split参数必须是'train'/'val'/'test'，当前为: {self.split}")
        self.samples = [all_samples[i] for i in selected]

        # 从JSON中获取各数据集的根目录信息
        with open(self.json_path, 'r') as f:
            data_info = json.load(f)
            self.datasets_info = data_info.get("datasets_info", {})
        # 不再全局保存raw_image_root_dir和mask_root_dir

        # 构建数据增强
        from utils.augmentation import build_augmentations
        self.augmentations = build_augmentations(
            mode='train' if self.split == 'train' else 'val',
            height=cfg_base.IMAGE_HEIGHT,
            width=cfg_base.IMAGE_WIDTH
        )
        print(f"成功从 {self.json_path} 解析数据。总样本: {num_samples}，本split({self.split})样本: {len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample_info = self.samples[idx]
        dataset = sample_info['dataset']
        raw_root = Path(self.datasets_info[dataset]['raw_image_root'])
        mask_root = Path(self.datasets_info[dataset]['mask_root'])
        # --- 1. 加载并堆叠3D图像 ---
        image_frames = []
        frame_paths = sample_info.get("frames", [])
        for frame_path in frame_paths:
            full_path = raw_root / frame_path
            img = np.array(Image.open(full_path).convert('L'), dtype=np.float32) / 255.0
            image_frames.append(img)
        stacked_image = np.stack(image_frames, axis=-1)
        # --- 2. 加载掩码 ---
        mask_path = mask_root / sample_info.get("mask_file")
        mask_img = Image.open(mask_path).convert('RGB')
        mask_np = np.array(mask_img)
        
        # --- 新增: 统一调整图像和掩码尺寸 ---
        target_height, target_width = cfg_base.IMAGE_HEIGHT, cfg_base.IMAGE_WIDTH
        
        # 调整3D图像的每个通道
        resized_frames = [cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA) for frame in image_frames]
        stacked_image = np.stack(resized_frames, axis=-1)

        # 调整掩码
        mask_np = cv2.resize(mask_np, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

        # --- 新增: 确保图像通道数符合模型要求 (3通道) ---
        if stacked_image.shape[-1] == 1:
            stacked_image = np.repeat(stacked_image, 3, axis=-1)
        
        mapping = cfg_base.MAPPING
        mask = np.zeros(mask_np.shape[:2], dtype=np.int64)
        for rgb, idx_map in mapping.items():
            mask[(mask_np == rgb).all(axis=-1)] = idx_map
        # --- 3. 应用数据增强 ---
        if self.augmentations:
            augmented = self.augmentations(image=stacked_image, mask=mask)
            image_tensor = augmented['image']
            mask_tensor = augmented['mask'].long()
        else:
            image_tensor = torch.from_numpy(stacked_image.transpose((2, 0, 1)))
            mask_tensor = torch.from_numpy(mask).long()
        
        # --- 4. 根据标志决定是否返回dataset名 ---
        if self.return_dataset_name:
            dataset_name = sample_info.get('dataset', 'unknown')
            return image_tensor, mask_tensor, dataset_name
        else:
            return image_tensor, mask_tensor