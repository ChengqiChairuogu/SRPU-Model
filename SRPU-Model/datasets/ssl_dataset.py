import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Optional, List, Callable, Dict, Any, Tuple
from pathlib import Path
import json
from PIL import Image

# 这是一个独立的 Dataset 类，用于自监督学习
# 它只加载图像，不加载掩码

class SSLDataset(Dataset):
    def __init__(self,
                 json_file_path: Path,
                 project_root: Path,
                 raw_image_root_name_in_json: str,
                 patch_size: int,
                 mask_ratio: float,
                 transform: Optional[Callable] = None):
        """
        用于自监督学习（掩码图像建模风格）的Dataset。

        Args:
            json_file_path (Path): 指向无标签数据集JSON文件的绝对路径。
            project_root (Path): 项目根目录的绝对路径。
            raw_image_root_name_in_json (str): 在JSON中记录的原始图像根目录名。
            patch_size (int): 将图像分割成的每个小块的尺寸。
            mask_ratio (float): 要遮挡掉的图像块的比例。
            transform (Optional[Callable]): 在遮挡之前应用于原始PIL图像的转换 (例如 Resize, ToTensor)。
        """
        self.project_root = project_root
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.transform = transform

        if not json_file_path.exists():
            raise FileNotFoundError(f"SSL用的JSON文件未找到: {json_file_path}")

        with open(json_file_path, 'r') as f:
            self.data_info = json.load(f)
        
        self.samples = self.data_info.get("samples", [])
        self.raw_image_root_name = self.data_info.get("root_raw_image_dir", raw_image_root_name_in_json)
        
        if not self.raw_image_root_name:
            raise ValueError(f"在JSON文件中未找到 'root_raw_image_dir': {json_file_path}")
        
        print(f"SSLDataset 初始化完成，共找到 {len(self.samples)} 个无标签样本。")


    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回:
            - masked_image_tensor (torch.Tensor): 被遮挡的图像，作为模型输入。
            - original_image_tensor (torch.Tensor): 原始未遮挡的图像，作为重建目标。
            - loss_mask (torch.Tensor): 布尔掩码，True表示该像素位置被遮挡，用于计算损失。
        """
        sample_info = self.samples[idx]
        # SSL通常在单帧2D图像上进行预训练。这里我们取序列的中心帧作为代表。
        # 如果您的序列中所有帧都相关，也可以随机选择一帧。
        center_frame_path_relative = sample_info["frames"][len(sample_info["frames"]) // 2]
        
        image_path = self.project_root / self.raw_image_root_name / center_frame_path_relative
        try:
            # 简化处理：加载为灰度图 (L)。如果您的SEM是多通道的，应按实际情况加载。
            original_image_pil = Image.open(image_path).convert('L')
        except Exception as e:
            raise IOError(f"打开图像失败 {image_path}: {e}")

        # 应用基础转换（如Resize, ToTensor）
        if self.transform:
            image_tensor = self.transform(original_image_pil)
        else:
            from torchvision import transforms
            image_tensor = transforms.ToTensor()(original_image_pil)

        # --- 执行随机块遮挡 (Random Patch Masking) ---
        _, h, w = image_tensor.shape
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError(f"图像尺寸 ({h}, {w}) 不能被 patch_size ({self.patch_size}) 整除。请在transform中加入Resize。")
        
        num_patches_h = h // self.patch_size
        num_patches_w = w // self.patch_size
        num_patches_total = num_patches_h * num_patches_w
        num_patches_to_mask = int(num_patches_total * self.mask_ratio)

        # 随机生成要遮挡的patch的索引
        patch_indices_shuffled = np.random.permutation(num_patches_total)
        masked_indices = patch_indices_shuffled[:num_patches_to_mask]

        # 创建一个布尔掩码 (True表示该patch被遮挡)
        patch_mask_flat = torch.zeros(num_patches_total, dtype=torch.bool)
        patch_mask_flat[masked_indices] = True
        patch_mask_2d = patch_mask_flat.reshape(num_patches_h, num_patches_w)

        # 创建被遮挡的图像
        masked_image_tensor = image_tensor.clone()
        # 将被遮挡的patch区域像素值设为0.5 (灰色)，或均值、随机值
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                if patch_mask_2d[i, j]:
                    h_start, w_start = i * self.patch_size, j * self.patch_size
                    masked_image_tensor[:, h_start:h_start+self.patch_size, w_start:w_start+self.patch_size] = 0.5

        # 将2D的patch掩码扩展回像素维度，用于计算损失
        loss_mask = patch_mask_2d.repeat_interleave(self.patch_size, dim=0).repeat_interleave(self.patch_size, dim=1)

        return masked_image_tensor, image_tensor, loss_mask