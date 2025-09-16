# utils/stratified_sample_util.py
from torch.utils.data import DataLoader, Sampler
import random
import numpy as np
import torch


class StratifiedBatchSampler(Sampler):
    def __init__(self, dataset_indices: dict[str, list[int]], batch_size: int, shuffle: bool = True):
        self.dataset_indices = dataset_indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.keys = list(dataset_indices.keys())
        # 每个数据集在一个 batch 中的样本数（向下取整，至少 1）
        self.per_dataset = max(1, batch_size // len(self.keys))

    def __iter__(self):
        if self.shuffle:
            for k in self.keys:
                random.shuffle(self.dataset_indices[k])

        min_len = min(len(v) for v in self.dataset_indices.values())
        batches = []
        for i in range(0, min_len, self.per_dataset):
            batch = []
            for k in self.keys:
                batch.extend(self.dataset_indices[k][i:i + self.per_dataset])
            # 如果 batch 不够大，随机补齐
            if len(batch) < self.batch_size:
                needed = self.batch_size - len(batch)
                extra = []
                for k in self.keys:
                    extra.extend(random.sample(self.dataset_indices[k], min(needed, len(self.dataset_indices[k]))))
                    if len(extra) >= needed:
                        break
                batch.extend(extra[:needed])
            batches.append(batch)
        return iter(batches)

    def __len__(self):
        return sum(len(v) for v in self.dataset_indices.values()) // self.batch_size


def create_stratified_dataloader(dataset, dataset_indices, batch_size, shuffle=True, seed=None, **kwargs):
    """
    创建分层采样的 DataLoader
    :param dataset: Dataset 对象
    :param dataset_indices: dict，每个 key 是子数据集名，值是样本索引列表
    :param batch_size: 每个 batch 样本总数
    :param shuffle: 是否打乱每个子数据集内部顺序
    :param seed: 随机种子（可选）
    :param kwargs: 传给 DataLoader 的其他参数（不能包含 batch_size/batch_sampler）
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    sampler = StratifiedBatchSampler(dataset_indices, batch_size, shuffle=shuffle)
    return DataLoader(dataset, batch_sampler=sampler, **kwargs)
