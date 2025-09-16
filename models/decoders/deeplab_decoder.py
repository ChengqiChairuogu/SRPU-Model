# 文件路径: models/decoders/deeplab_decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F

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

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        ])
        for rate in rates:
            self.convs.append(
                nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
            )
        self.image_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.project = nn.Sequential(nn.Conv2d(len(self.convs) * out_channels + out_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Dropout(0.5))

    def forward(self, x):
        res = [conv(x) for conv in self.convs]
        pool_feat = F.interpolate(self.image_pool(x), size=ensure_interpolate_size(x.shape[2:]), mode='bilinear', align_corners=False)
        res.append(pool_feat)
        return self.project(torch.cat(res, dim=1))

class DeepLabV3PlusDecoder(nn.Module):
    def __init__(self, encoder_channels, num_classes):
        """
        参数:
        - encoder_channels (list): 编码器输出的各阶段特征图的通道数，从浅到深。
        """
        super().__init__()
        high_level_channels = encoder_channels[-1]
        low_level_channels = encoder_channels[1] # ResNet: stage1, EfficienNet: stage1

        self.aspp = ASPP(high_level_channels, 256, [6, 12, 18])
        self.low_level_conv = nn.Sequential(nn.Conv2d(low_level_channels, 48, 1, bias=False), nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        self.final_conv = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, features):
        high_level_feat = features[-1]
        low_level_feat = features[1]
        
        x = self.aspp(high_level_feat)
        low_level_feat = self.low_level_conv(low_level_feat)
        
        x = F.interpolate(x, size=ensure_interpolate_size(low_level_feat.shape[2:]), mode='bilinear', align_corners=False)
        x = torch.cat([x, low_level_feat], dim=1)
        x = self.final_conv(x)
        
        # 将输出上采样到原始尺寸 (如果需要)
        # x = F.interpolate(x, size=original_size, mode='bilinear', align_corners=False)
        return x