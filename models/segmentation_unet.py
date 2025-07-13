import torch
import torch.nn as nn
from typing import List

class SegmentationUNet(nn.Module):
    """
    一个通用的分割模型容器。
    它将一个编码器和一个解码器组合成一个端到端的模型。
    """
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        定义数据的前向传播路径：输入 -> 编码器 -> 解码器 -> 输出。
        """
        # 编码器提取特征，通常返回一个包含多个层级特征的列表
        encoder_features = self.encoder(x)
        
        # 解码器利用这些特征来生成最终的分割图
        # 解码器内部会处理如何使用这个特征列表
        segmentation_map = self.decoder(encoder_features)
        
        return segmentation_map