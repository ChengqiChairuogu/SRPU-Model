# SRPU-Model/datasets/stratified_sampler.py
"""
分层采样器，确保每个batch都包含来自不同数据集的样本
解决数据不平衡问题，特别是dataset3_LNOCl样本较少的情况
"""

import torch
from torch.utils.data import Sampler
from typing import List, Dict, Iterator
import random
import numpy as np
from collections import defaultdict


class StratifiedBatchSampler(Sampler):
    """
    分层批次采样器
    
    确保每个batch都包含来自不同数据集的样本，解决数据不平衡问题
    """
    
    def __init__(self, 
                 dataset_indices: Dict[str, List[int]], 
                 batch_size: int,
                 shuffle: bool = True,
                 seed: int = 42,
                 min_samples_per_dataset: int = 1):
        """
        初始化分层采样器
        
        Args:
            dataset_indices: 字典，键为数据集名称，值为该数据集样本的索引列表
            batch_size: 批次大小
            shuffle: 是否打乱数据
            seed: 随机种子
            min_samples_per_dataset: 每个batch中每个数据集的最少样本数
        """
        self.dataset_indices = dataset_indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.min_samples_per_dataset = min_samples_per_dataset
        
        # 计算每个数据集在每个batch中的目标样本数
        self.dataset_names = list(dataset_indices.keys())
        self.num_datasets = len(self.dataset_names)
        
        # 确保batch_size足够大以容纳所有数据集
        min_required_batch_size = self.num_datasets * self.min_samples_per_dataset
        if self.batch_size < min_required_batch_size:
            print(f"警告: batch_size ({self.batch_size}) 小于最小要求 ({min_required_batch_size})")
            print(f"将自动调整batch_size为 {min_required_batch_size}")
            self.batch_size = min_required_batch_size
        
        # 计算每个数据集在每个batch中的样本数
        self.samples_per_dataset = self._calculate_samples_per_dataset()
        
        # 设置随机种子
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        print(f"分层采样器初始化完成:")
        print(f"  数据集: {self.dataset_names}")
        print(f"  每个batch中各数据集样本数: {self.samples_per_dataset}")
        print(f"  总batch大小: {self.batch_size}")
    
    def _calculate_samples_per_dataset(self) -> Dict[str, int]:
        """计算每个数据集在每个batch中的样本数"""
        samples_per_dataset = {}
        
        # 首先为每个数据集分配最小样本数
        remaining_samples = self.batch_size - (self.num_datasets * self.min_samples_per_dataset)
        
        # 按数据集样本数量比例分配剩余样本
        total_samples = sum(len(indices) for indices in self.dataset_indices.values())
        
        for dataset_name, indices in self.dataset_indices.items():
            base_samples = self.min_samples_per_dataset
            if remaining_samples > 0:
                # 按比例分配剩余样本
                proportional_samples = int(remaining_samples * len(indices) / total_samples)
                samples_per_dataset[dataset_name] = base_samples + proportional_samples
            else:
                samples_per_dataset[dataset_name] = base_samples
        
        return samples_per_dataset
    
    def __iter__(self) -> Iterator[List[int]]:
        """生成批次索引"""
        # 为每个数据集创建索引池
        dataset_pools = {}
        for dataset_name, indices in self.dataset_indices.items():
            pool = indices.copy()
            if self.shuffle:
                random.shuffle(pool)
            dataset_pools[dataset_name] = pool
        
        # 生成批次
        while True:
            batch_indices = []
            
            # 从每个数据集采样指定数量的样本
            for dataset_name, target_count in self.samples_per_dataset.items():
                pool = dataset_pools[dataset_name]
                
                # 如果池子空了，重新填充并打乱
                if len(pool) < target_count:
                    pool.extend(self.dataset_indices[dataset_name].copy())
                    if self.shuffle:
                        random.shuffle(pool)
                
                # 采样指定数量的样本
                sampled_indices = pool[:target_count]
                batch_indices.extend(sampled_indices)
                
                # 更新池子
                dataset_pools[dataset_name] = pool[target_count:]
            
            # 如果某个数据集没有足够样本，停止生成
            if len(batch_indices) < self.batch_size:
                break
            
            yield batch_indices
    
    def __len__(self) -> int:
        """计算总批次数"""
        # 计算最少的完整batch数
        min_batches = float('inf')
        for dataset_name, indices in self.dataset_indices.items():
            target_count = self.samples_per_dataset[dataset_name]
            batches_for_dataset = len(indices) // target_count
            min_batches = min(min_batches, batches_for_dataset)
        
        return int(min_batches)


def create_stratified_dataloader(dataset, 
                                dataset_indices: Dict[str, List[int]],
                                batch_size: int,
                                shuffle: bool = True,
                                seed: int = 42,
                                num_workers: int = 2,
                                pin_memory: bool = True,
                                **kwargs):
    """
    创建分层采样的数据加载器
    
    Args:
        dataset: 数据集对象
        dataset_indices: 字典，键为数据集名称，值为该数据集样本的索引列表
        batch_size: 批次大小
        shuffle: 是否打乱数据
        seed: 随机种子
        num_workers: 工作进程数
        pin_memory: 是否使用pin_memory
        **kwargs: 其他DataLoader参数
    
    Returns:
        DataLoader: 分层采样的数据加载器
    """
    from torch.utils.data import DataLoader
    
    # 创建分层采样器
    sampler = StratifiedBatchSampler(
        dataset_indices=dataset_indices,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **kwargs
    )
    
    return dataloader


def get_dataset_indices_from_sem_dataset(dataset) -> Dict[str, List[int]]:
    """
    从SemSegmentationDataset中提取数据集索引
    
    Args:
        dataset: SemSegmentationDataset实例
    
    Returns:
        Dict[str, List[int]]: 数据集名称到索引列表的映射
    """
    dataset_indices = defaultdict(list)
    
    for idx, sample in enumerate(dataset.samples):
        dataset_name = sample.get('dataset', 'unknown')
        dataset_indices[dataset_name].append(idx)
    
    return dict(dataset_indices) 