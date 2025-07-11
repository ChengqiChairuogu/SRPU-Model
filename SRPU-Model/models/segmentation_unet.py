# self_segmentation/models/segmentation_unet.py

import torch
import torch.nn as nn
from typing import Dict, List

# 导入您自己定义的编码器和解码器模块
from .encoders.unet_encoder import UNetEncoder
from .decoders.unet_decoder import UNetDecoder

class SegmentationUNet(nn.Module):
    def __init__(self,
                 encoder: UNetEncoder,
                 decoder: UNetDecoder):
        """
        一个由独立的编码器和解码器模块组成的U-Net模型。

        Args:
            encoder (UNetEncoder): UNet编码器实例。
            decoder (UNetDecoder): UNet解码器实例。
        """
        super().__init__()
        # 关键：将编码器和解码器作为可访问的子模块保存
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        模型的前向传播。

        Args:
            x (torch.Tensor): 输入图像张量 (B, C_in, H, W)。

        Returns:
            torch.Tensor: 分割图的logits输出 (B, C_out, H, W)。
        """
        # 1. 编码器提取特征
        # UNetEncoder 的 forward 方法返回一个包含各阶段特征的列表
        # 列表顺序是从浅层到深层
        encoder_features: List[torch.Tensor] = self.encoder(x)
        
        # 2. 解码器利用这些特征进行重建
        # UNetDecoder 的 forward 方法接收最深层的特征和用于跳跃连接的浅层特征
        output_logits: torch.Tensor = self.decoder(encoder_features)
        
        return output_logits
