import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

# 导入通用的形状处理函数
try:
    from utils.training_util import ensure_interpolate_size
except ImportError:
    # 如果无法导入，定义一个本地版本
    def ensure_interpolate_size(target_shape):
        """
        确保interpolate的size参数格式正确
        Args:
            target_shape: 目标形状，可能是torch.Size([H, W])或torch.Size([H])等
        Returns:
            正确格式的size参数，确保是(H, W)格式
        """
        if len(target_shape) == 1:
            # 如果target_shape是单个数字，转换为(H, W)格式
            return (target_shape[0], target_shape[0])
        elif len(target_shape) == 2:
            # 如果已经是(H, W)格式，直接返回
            return target_shape
        else:
            # 如果长度大于2，取最后两个维度
            return target_shape[-2:]

class MAEDecoder(nn.Module):
    """
    一个简化的解码器，用于从U-Net编码器最深层的特征图中重建原始图像。
    """
    def __init__(self, encoder_channels: Tuple[int, ...], decoder_embed_dim: int, out_channels: int = 1):
        super().__init__()
        in_channels = encoder_channels[-1]
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, decoder_embed_dim, kernel_size=1),
            nn.GELU(),
            nn.ConvTranspose2d(decoder_embed_dim, decoder_embed_dim // 2, kernel_size=4, stride=4, padding=0),
            nn.GELU(),
            nn.ConvTranspose2d(decoder_embed_dim // 2, decoder_embed_dim // 4, kernel_size=4, stride=4, padding=0),
            nn.GELU(),
            nn.Conv2d(decoder_embed_dim // 4, out_channels, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        reconstructed = self.decoder(features)
        return reconstructed

class MaskedAutoencoderUNet(nn.Module):
    """
    将U-Net编码器与一个简单的MAE解码器结合，用于自监督学习。
    """
    def __init__(self,
                 encoder: nn.Module,
                 decoder_embed_dim: int,
                 n_channels_in: int = 1):
        super().__init__()
        self.encoder = encoder
        encoder_channels = self.encoder.get_channels()
        self.decoder = MAEDecoder(
            encoder_channels=encoder_channels,
            decoder_embed_dim=decoder_embed_dim,
            out_channels=n_channels_in
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoder_features_list = self.encoder(x)
        latent_representation = encoder_features_list[-1]
        reconstructed_image = self.decoder(latent_representation)

        if reconstructed_image.shape[-2:] != x.shape[-2:]:
            target_shape = ensure_interpolate_size(x.shape[-2:])
            reconstructed_image = F.interpolate(
                reconstructed_image,
                size=target_shape,
                mode='bilinear',
                align_corners=False
            )
            
        return reconstructed_image