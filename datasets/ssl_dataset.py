# self_segmentation/datasets/ssl_dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple
from pathlib import Path
import json
from PIL import Image
import sys

# --- 导入 ---
try:
    from configs import base as cfg_base
    from configs.selfup import ssl_config as cfg_ssl
    # **关键修正 1**: 导入我们统一的增强函数
    from utils.augmentation_util import build_augmentations
except ImportError:
    if __name__ == '__main__' and __package__ is None:
        current_script_path = Path(__file__).resolve()
        project_root_for_import = current_script_path.parent.parent
        sys.path.insert(0, str(project_root_for_import))
        from configs import base as cfg_base
        from configs.selfup import ssl_config as cfg_ssl
        from utils.augmentation_util import build_augmentations
    else:
        raise

class SSLDataset(Dataset):
    """用于自监督学习（如MAE）的数据集。"""
    def __init__(self,
                 json_file_path: Path,
                 project_root: Path):
        
        self.project_root = project_root
        # self.patch_size = cfg_ssl.PATCH_SIZE
        self.patch_size = 16  # 适配UNet自编码器，无需从config读取
        self.mask_ratio = cfg_ssl.MASK_RATIO

        if not json_file_path.exists():
            raise FileNotFoundError(f"SSL JSON文件未找到: {json_file_path}")

        with open(json_file_path, 'r') as f:
            self.data_info = json.load(f)
        
        self.samples = self.data_info.get("samples", [])
        
        # 获取datasets_info，用于构建每个样本的完整路径
        self.datasets_info = self.data_info.get("datasets_info", {})
        
        if not self.datasets_info:
            raise ValueError("JSON缺少datasets_info字段")
        
        print(f"检测到清晰度平均化SSL JSON格式，包含 {len(self.datasets_info)} 个数据集")
        for dataset_name, dataset_path in self.datasets_info.items():
            print(f"  - {dataset_name}: {dataset_path}")
        
        # **关键修正 2**: 调用 build_augmentations 来创建几何变换流水线
        # 模式为 "ssl"，并且不在此处进行归一化，因为MAE是在patching之后处理
        self.transform = build_augmentations(
            mode="ssl",
            height=cfg_base.IMAGE_HEIGHT,
            width=cfg_base.IMAGE_WIDTH,
            use_normalization=False
        )

        print(f"SSLDataset 初始化完成，共找到 {len(self.samples)} 个无标签样本。")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample_info = self.samples[idx]
        
        center_frame_path = None

        if "frames" in sample_info:
            center_frame = sample_info["frames"][len(sample_info["frames"]) // 2]
            center_frame_path = self.raw_image_root / center_frame
        elif "image_file" in sample_info:
            image_file_path = sample_info["image_file"]
            dataset_name = sample_info.get("dataset")
            
            if dataset_name and dataset_name in self.datasets_info:
                # **修复路径重复问题**: 直接使用image_file_path，因为它已经包含了完整路径
                center_frame_path = self.project_root / image_file_path
                
                # 调试信息（可选）
                # print(f"构建路径: {self.project_root} / {image_file_path} = {center_frame_path}")
            else:
                raise ValueError(f"样本中缺少dataset字段或dataset '{dataset_name}' 不在datasets_info中: {sample_info}")
        else:
            raise KeyError("样本中既没有 'frames' 也没有 'image_file' 字段！")

        if not center_frame_path:
            raise ValueError(f"无法为样本构建有效路径: {sample_info}")
        
        try:
            # 加载为灰度图，并保持为 NumPy 数组以用于 albumentations
            original_image_np = np.array(Image.open(center_frame_path).convert('L'))
        except Exception as e:
            raise IOError(f"打开图像失败 {center_frame_path}: {e}")

        # 1. 应用几何数据增强（裁剪、翻转、旋转等）
        augmented = self.transform(image=original_image_np)
        # 将图像转换为 [0, 1] 范围的 float32 张量
        image_tensor = augmented['image'].float() / 255.0 # 形状: (1, H, W)

        # 2. 执行随机块遮挡 (Random Patch Masking for MAE-style pre-training)
        _, h, w = image_tensor.shape
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError(f"图像尺寸 ({h}, {w}) 不能被 patch_size ({self.patch_size}) 整除。")
        
        num_patches_h = h // self.patch_size
        num_patches_w = w // self.patch_size
        num_patches_total = num_patches_h * num_patches_w
        num_patches_to_mask = int(num_patches_total * self.mask_ratio)

        # 创建并打乱 patch 索引
        patch_indices_shuffled = np.random.permutation(num_patches_total)
        masked_indices = patch_indices_shuffled[:num_patches_to_mask]

        # 创建2D的 patch 掩码
        patch_mask_flat = torch.zeros(num_patches_total, dtype=torch.bool)
        patch_mask_flat[masked_indices] = True
        patch_mask_2d = patch_mask_flat.reshape(num_patches_h, num_patches_w)

        # 创建被遮挡的图像
        masked_image_tensor = image_tensor.clone()
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                if patch_mask_2d[i, j]:
                    h_start, w_start = i * self.patch_size, j * self.patch_size
                    # 使用一个固定值（例如0.5，近似灰度均值）进行遮挡
                    masked_image_tensor[:, h_start:h_start+self.patch_size, w_start:w_start+self.patch_size] = 0.5 

        # 创建用于计算损失的像素级掩码
        loss_mask = patch_mask_2d.repeat_interleave(self.patch_size, dim=0).repeat_interleave(self.patch_size, dim=1)

        # 返回 被遮挡的图像、原始图像、损失掩码
        return masked_image_tensor, image_tensor, loss_mask