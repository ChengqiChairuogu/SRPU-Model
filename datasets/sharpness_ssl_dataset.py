"""清晰度自监督训练数据集"""

import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from utils.image_sharpness_util import build_degrader, set_seed
from configs.image_sharpness.sharpness_train_config import *


class SharpnessSSLDataset(Dataset):
    """清晰度自监督训练数据集
    
    该数据集用于训练图像清晰度恢复模型：
    - 输入：经过降质处理的低清晰度图像
    - 目标：原始高清晰度图像
    - 支持多种降质模式和不同清晰度梯度
    """
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        gradient_type: str = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        max_depth: int = 10,
        use_cache: bool = True
    ):
        """
        Args:
            root_dir: 原始图像根目录
            gradient_type: 清晰度梯度类型 ("low", "medium", "high")
            transform: 输入图像变换
            target_transform: 目标图像变换
            max_depth: 递归深度上限
            use_cache: 是否使用缓存
        """
        super().__init__()
        
        # 设置随机种子
        set_seed(RANDOM_SEED)
        
        self.root_dir = Path(root_dir)
        self.gradient_type = gradient_type
        self.transform = transform
        self.target_transform = target_transform
        self.use_cache = use_cache
        
        # 设置默认梯度类型
        self.gradient_type = gradient_type or CURRENT_GRADIENT_TYPE
        
        # 验证梯度类型
        if self.gradient_type not in SHARPNESS_GRADIENTS:
            raise ValueError(f"不支持的梯度类型: {self.gradient_type}")
        
        # 获取梯度配置
        self.gradient_config = SHARPNESS_GRADIENTS[self.gradient_type]
        
        # 收集图像路径
        self.image_paths = self._collect_image_paths(max_depth)
        
        # 初始化降质器
        self.degraders = self._initialize_degraders()
        
        # 初始化变换
        self._initialize_transforms()
        
        # 缓存
        self.cache = {} if use_cache else None
        
        print(f"初始化清晰度SSL数据集: {len(self.image_paths)} 张图像, 梯度类型: {self.gradient_type} ({self.gradient_config['name']})")
    
    def _collect_image_paths(self, max_depth: int) -> List[Path]:
        """收集图像路径"""
        if not self.root_dir.exists():
            raise FileNotFoundError(f"数据集根目录不存在: {self.root_dir}")
        
        image_paths = []
        
        def _walk(cur: Path, depth: int):
            if depth > max_depth:
                return
            
            # 跳过隐藏目录
            if cur.name.startswith("."):
                return
            
            for item in cur.iterdir():
                if item.is_dir():
                    _walk(item, depth + 1)
                elif item.is_file() and item.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".bmp"]:
                    image_paths.append(item)
        
        _walk(self.root_dir, 0)
        
        # 去重并排序
        image_paths = sorted(list(set(image_paths)))
        
        if len(image_paths) == 0:
            raise RuntimeError(f"在 {self.root_dir} 中没有找到有效的图像文件")
        
        return image_paths
    
    def _initialize_degraders(self) -> List[Callable]:
        """初始化降质器"""
        degraders = []
        
        for mode in self.gradient_config["degradation_modes"]:
            params = self.gradient_config["degradation_params"].get(mode, {})
            degrader = build_degrader(mode, **params)
            degraders.append(degrader)
        
        return degraders
    
    def _initialize_transforms(self):
        """初始化图像变换"""
        # 基础变换
        self.base_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
        ])
        
        # 数据增强变换
        if AUGMENTATION_CONFIG["horizontal_flip"]:
            self.aug_transform = transforms.RandomHorizontalFlip(p=0.5)
        else:
            self.aug_transform = transforms.Lambda(lambda x: x)
        
        # 颜色增强
        color_transforms = []
        if AUGMENTATION_CONFIG["brightness"] > 0:
            color_transforms.append(
                transforms.ColorJitter(brightness=AUGMENTATION_CONFIG["brightness"])
            )
        if AUGMENTATION_CONFIG["contrast"] > 0:
            color_transforms.append(
                transforms.ColorJitter(contrast=AUGMENTATION_CONFIG["contrast"])
            )
        if AUGMENTATION_CONFIG["saturation"] > 0:
            color_transforms.append(
                transforms.ColorJitter(saturation=AUGMENTATION_CONFIG["saturation"])
            )
        if AUGMENTATION_CONFIG["hue"] > 0:
            color_transforms.append(
                transforms.ColorJitter(hue=AUGMENTATION_CONFIG["hue"])
            )
        
        if color_transforms:
            self.color_transform = transforms.Compose(color_transforms)
        else:
            self.color_transform = transforms.Lambda(lambda x: x)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取数据项
        
        Returns:
            tuple: (degraded_image, original_image)
        """
        image_path = self.image_paths[index]
        
        # 检查缓存
        if self.use_cache and index in self.cache:
            return self.cache[index]
        
        try:
            # 加载原始图像
            with Image.open(image_path) as img:
                img = img.convert("RGB")
            
            # 应用基础变换
            original_tensor = self.base_transform(img)
            
            # 应用数据增强
            original_tensor = self.aug_transform(original_tensor)
            original_tensor = self.color_transform(original_tensor)
            
            # 随机选择降质器
            degrader = random.choice(self.degraders)
            
            # 应用降质
            degraded_tensor = degrader(original_tensor.unsqueeze(0)).squeeze(0)
            
            # 确保值在有效范围内
            degraded_tensor = torch.clamp(degraded_tensor, 0, 1)
            
            # 数据验证：检查降质效果
            if random.random() < DATA_VALIDATION_FREQUENCY:  # 使用配置的概率
                original_max = original_tensor.max().item()
                original_min = original_tensor.min().item()
                degraded_max = degraded_tensor.max().item()
                degraded_min = degraded_tensor.min().item()
                
                # 计算降质前后的差异
                max_diff = max(abs(original_max - degraded_max), abs(original_min - degraded_min))
                mean_diff = abs(original_tensor.mean() - degraded_tensor.mean()).item()
                var_diff = abs(original_tensor.var() - degraded_tensor.var()).item()
                
                # 检查降质是否合理 - 使用更严格的阈值
                if max_diff < 0.02 and mean_diff < 0.005 and var_diff < 0.005:
                    print(f"警告: 图像 {image_path.name} 降质效果可能不明显")
                    print(f"  最大差异: {max_diff:.3f}, 均值差异: {mean_diff:.3f}, 方差差异: {var_diff:.3f}")
                    print(f"  降质器: {degrader.__name__ if hasattr(degrader, '__name__') else 'unknown'}")
                    
                    # 如果降质效果不明显，尝试增强降质
                    if "gaussian_blur" in str(degrader):
                        print(f"  尝试增强高斯模糊效果...")
                        # 应用更强的模糊
                        import torch.nn.functional as F
                        kernel_size = 15
                        sigma = 3.0
                        
                        # 创建更强的2D高斯核
                        x = torch.arange(-(kernel_size//2), kernel_size//2 + 1, dtype=torch.float32)
                        y = torch.arange(-(kernel_size//2), kernel_size//2 + 1, dtype=torch.float32)
                        x_grid, y_grid = torch.meshgrid(x, y, indexing='ij')
                        
                        kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
                        kernel = kernel / kernel.sum()
                        
                        device = degraded_tensor.device
                        kernel = kernel.to(device)
                        kernel = kernel.view(1, 1, kernel_size, kernel_size)
                        kernel = kernel.repeat(3, 1, 1, 1)
                        
                        padding = kernel_size // 2
                        degraded_tensor = F.conv2d(degraded_tensor.unsqueeze(0), kernel, padding=padding, groups=3).squeeze(0)
                        print(f"  已应用增强模糊，新差异: 均值={abs(original_tensor.mean() - degraded_tensor.mean()):.3f}")
                    
                    elif "bicubic_downup" in str(degrader):
                        print(f"  尝试增强缩放降质效果...")
                        # 应用更强的缩放
                        import torch.nn.functional as F
                        scale = 0.2  # 更小的缩放比例
                        B, C, H, W = original_tensor.unsqueeze(0).shape
                        new_H, new_W = int(H * scale), int(W * scale)
                        
                        downsampled = F.interpolate(original_tensor.unsqueeze(0), size=(new_H, new_W), mode='bicubic', align_corners=False)
                        degraded_tensor = F.interpolate(downsampled, size=(H, W), mode='bicubic', align_corners=False).squeeze(0)
                        print(f"  已应用增强缩放，新差异: 均值={abs(original_tensor.mean() - degraded_tensor.mean()):.3f}")
                    
                    elif "jpeg_compression" in str(degrader):
                        print(f"  尝试增强JPEG压缩效果...")
                        # 应用更强的JPEG压缩
                        import io
                        
                        img_tensor = original_tensor
                        if img_tensor.max() <= 1.0:
                            img_tensor = (img_tensor * 255).clamp(0, 255).byte()
                        
                        img_pil = Image.fromarray(img_tensor.permute(1, 2, 0).cpu().numpy())
                        
                        buffer = io.BytesIO()
                        img_pil.save(buffer, format='JPEG', quality=15)  # 更低的JPEG质量
                        buffer.seek(0)
                        
                        img_degraded = Image.open(buffer)
                        degraded_tensor = torch.from_numpy(np.array(img_degraded)).float() / 255.0
                        degraded_tensor = degraded_tensor.permute(2, 0, 1)
                        print(f"  已应用增强JPEG压缩，新差异: 均值={abs(original_tensor.mean() - degraded_tensor.mean()):.3f}")
                
                # 检查是否有异常值
                if torch.isnan(degraded_tensor).any() or torch.isinf(degraded_tensor).any():
                    print(f"警告: 图像 {image_path.name} 降质后包含异常值")
                    # 使用原始图像作为降质图像
                    degraded_tensor = original_tensor.clone()
            
            # 应用目标变换（如果有）
            if self.target_transform:
                original_tensor = self.target_transform(original_tensor)
            
            result = (degraded_tensor, original_tensor)
            
            # 缓存结果
            if self.use_cache:
                self.cache[index] = result
            
            return result
            
        except Exception as e:
            print(f"加载图像 {image_path} 失败: {e}")
            # 返回一个默认的样本
            dummy_tensor = torch.zeros(3, *IMAGE_SIZE)
            return (dummy_tensor, dummy_tensor)
    
    def get_gradient_info(self) -> Dict:
        """获取梯度信息"""
        return {
            "type": self.gradient_type,
            "name": self.gradient_config["name"],
            "degradation_modes": self.gradient_config["degradation_modes"],
            "target_improvement": self.gradient_config["target_improvement"]
        }


class SharpnessSSLDataLoader:
    """清晰度SSL数据加载器管理器"""
    
    def __init__(self, batch_size: int = None, num_workers: int = None):
        """
        Args:
            batch_size: 批处理大小
            num_workers: 工作进程数
        """
        self.batch_size = batch_size or BATCH_SIZE
        self.num_workers = num_workers or NUM_WORKERS
        
        # 创建数据集
        self.datasets = {}
        self.dataloaders = {}
        
        self._create_datasets()
        self._create_dataloaders()
    
    def _create_datasets(self):
        """创建不同梯度的数据集"""
        for dataset_name, dataset_path in DATASET_PATHS.items():
            if dataset_path.exists():
                for gradient_type in SHARPNESS_GRADIENTS.keys():
                    dataset_key = f"{dataset_name}_{gradient_type}"
                    
                    # 创建训练集
                    train_dataset = SharpnessSSLDataset(
                        root_dir=dataset_path,
                        gradient_type=gradient_type
                    )
                    
                    # 分割训练集和验证集
                    train_size = int(TRAIN_RATIO * len(train_dataset))
                    val_size = len(train_dataset) - train_size
                    
                    train_subset, val_subset = torch.utils.data.random_split(
                        train_dataset, [train_size, val_size]
                    )
                    
                    self.datasets[f"{dataset_key}_train"] = train_subset
                    self.datasets[f"{dataset_key}_val"] = val_subset
                    
                    print(f"创建数据集: {dataset_key} - 训练: {len(train_subset)}, 验证: {len(val_subset)}")
    
    def _create_dataloaders(self):
        """创建数据加载器"""
        for dataset_key, dataset in self.datasets.items():
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=dataset_key.endswith("_train"),
                num_workers=self.num_workers,
                pin_memory=PIN_MEMORY,
                drop_last=dataset_key.endswith("_train")
            )
            
            self.dataloaders[dataset_key] = dataloader
    
    def get_dataloader(self, dataset_key: str) -> torch.utils.data.DataLoader:
        """获取指定的数据加载器"""
        if dataset_key not in self.dataloaders:
            raise KeyError(f"数据集键不存在: {dataset_key}")
        return self.dataloaders[dataset_key]
    
    def get_all_dataloaders(self) -> Dict[str, torch.utils.data.DataLoader]:
        """获取所有数据加载器"""
        return self.dataloaders
    
    def get_dataset_info(self) -> Dict:
        """获取数据集信息"""
        info = {}
        for dataset_key, dataset in self.datasets.items():
            info[dataset_key] = {
                "size": len(dataset),
                "gradient_type": dataset_key.split("_")[-1] if "_" in dataset_key else "unknown"
            }
        return info


def create_sample_batch(dataset: SharpnessSSLDataset, num_samples: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
    """创建样本批次用于可视化"""
    degraded_samples = []
    original_samples = []
    
    for i in range(min(num_samples, len(dataset))):
        degraded, original = dataset[i]
        degraded_samples.append(degraded)
        original_samples.append(original)
    
    return torch.stack(degraded_samples), torch.stack(original_samples)


if __name__ == "__main__":
    # 测试数据集
    dataset = SharpnessSSLDataset(
        root_dir="datasets/dataset1_LInCl/raw_images"
        # gradient_type 将从配置文件读取默认值
    )
    
    print(f"数据集大小: {len(dataset)}")
    print(f"梯度信息: {dataset.get_gradient_info()}")
    
    # 测试数据加载
    degraded, original = dataset[0]
    print(f"降质图像形状: {degraded.shape}")
    print(f"原始图像形状: {original.shape}")
    
    # 测试数据加载器管理器
    loader_manager = SharpnessSSLDataLoader(batch_size=4)
    print(f"数据加载器信息: {loader_manager.get_dataset_info()}")
