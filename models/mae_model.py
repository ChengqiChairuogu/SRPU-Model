# self_segmentation/models/mae_model.py

import torch
import torch.nn as nn
from typing import Any

class MaskedAutoencoder(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 encoder_out_channels: int,
                 decoder_embed_dim: int,
                 n_channels_in: int,
                 image_size: int,
                 patch_size: int):
        """
        一个简化的掩码自编码器模型。

        Args:
            encoder (nn.Module): 一个预定义的编码器实例 (例如，来自 models/encoders/ 的U-Net编码器)。
            encoder_out_channels (int): 编码器输出特征图的通道数。
            decoder_embed_dim (int): 解码器中间层的嵌入维度。
            n_channels_in (int): 原始输入图像的通道数 (例如，灰度图为1)。
            image_size (int): 输入图像的边长 (假设为正方形)。
            patch_size (int): 图像块的尺寸。
        """
        super().__init__()
        self.encoder = encoder
        self.patch_size = patch_size
        
        # 计算编码器输出特征图的尺寸
        # 这是一个简化的假设，您需要根据您的编码器实际的下采样率来调整
        # 例如，如果编码器进行了3次2x2的最大池化，尺寸会变为 H / (2^3) = H / 8
        encoder_downsample_ratio = 8 # 示例，您需要根据您的编码器修改
        feature_map_size = image_size // encoder_downsample_ratio

        # 定义一个轻量级的解码器来重建图像
        # 它的任务是从编码器提取的深层特征恢复原始像素值
        # 这里使用转置卷积 (ConvTranspose2d) 来逐步上采样
        self.decoder = nn.Sequential(
            nn.Conv2d(encoder_out_channels, decoder_embed_dim, kernel_size=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(decoder_embed_dim, decoder_embed_dim // 2, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(decoder_embed_dim // 2, decoder_embed_dim // 4, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(decoder_embed_dim // 4, decoder_embed_dim // 8, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            
            # 最终输出层，通道数与原始输入图像一致
            nn.Conv2d(decoder_embed_dim // 8, n_channels_in, kernel_size=1),
            nn.Sigmoid() # 将输出值限制在 [0, 1] 范围，因为输入图像被归一化到此范围
        )

    def forward(self, x_masked: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x_masked (torch.Tensor): (B, C, H, W) 形状的被遮挡图像。

        Returns:
            torch.Tensor: (B, C, H, W) 形状的重建图像。
        """
        # 1. 使用编码器提取可见部分的特征
        features = self.encoder(x_masked)
        
        # 2. 使用解码器从特征重建整个图像
        reconstructed_image = self.decoder(features)
        
        return reconstructed_image
