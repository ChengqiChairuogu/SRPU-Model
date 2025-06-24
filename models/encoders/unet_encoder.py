# 文件路径: models/encoders/unet_encoder.py

import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(卷积 => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNetEncoder(nn.Module):
    def __init__(self, in_channels=3, base_c=64):
        """
        初始化经典的U-Net编码器。
        
        参数:
        - in_channels (int): 输入图像的通道数。
        - base_c (int): U-Net第一层的通道数，后续层会基于此进行扩展。
        """
        super().__init__()
        self.in_channels = in_channels
        self.base_c = base_c
        
        # 定义编码器的各个阶段
        self.inc = DoubleConv(in_channels, base_c)
        self.down1 = DoubleConv(base_c, base_c * 2)
        self.down2 = DoubleConv(base_c * 2, base_c * 4)
        self.down3 = DoubleConv(base_c * 4, base_c * 8)
        self.down4 = DoubleConv(base_c * 8, base_c * 16)
        
        # 定义下采样（池化）层
        self.pool = nn.MaxPool2d(2)

    def get_channels(self):
        """返回每个阶段输出的通道数，用于解码器配置"""
        return [self.base_c, self.base_c * 2, self.base_c * 4, self.base_c * 8, self.base_c * 16]

    def forward(self, x):
        """
        前向传播，返回一个包含各阶段特征图的列表，用于跳跃连接。
        """
        features = []
        
        # Stage 0
        x1 = self.inc(x)
        features.append(x1)
        
        # Stage 1
        x2 = self.pool(x1)
        x2 = self.down1(x2)
        features.append(x2)
        
        # Stage 2
        x3 = self.pool(x2)
        x3 = self.down2(x3)
        features.append(x3)
        
        # Stage 3
        x4 = self.pool(x3)
        x4 = self.down3(x4)
        features.append(x4)
        
        # Stage 4 (最深层)
        x5 = self.pool(x4)
        x5 = self.down4(x5)
        features.append(x5)
        
        return features