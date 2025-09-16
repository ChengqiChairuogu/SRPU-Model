import torch
import torch.nn as nn
from .encoders.unet_encoder import UNetEncoder
from .decoders.unet_decoder import UNetDecoder

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

class UNetAutoencoder(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super().__init__()
        self.encoder = UNetEncoder(n_channels=n_channels)
        encoder_channels = [64, 128, 256, 512, 1024, 2048]
        self.decoder = UNetDecoder(encoder_channels=encoder_channels, n_classes=n_classes)

    def forward(self, x):
        features = self.encoder(x)
        out = self.decoder(features)
        # 自动调整输出尺寸以匹配输入
        if out.shape[-2:] != x.shape[-2:]:
            target_shape = ensure_interpolate_size(x.shape[-2:])
            out = nn.functional.interpolate(out, size=target_shape, mode='bilinear', align_corners=False)
        return out 