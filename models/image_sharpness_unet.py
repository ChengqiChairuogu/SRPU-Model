#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
图像清晰度任务的深度UNet模型
- 增强模型容量，提高特征学习能力
- 专门用于自监督图像清晰度恢复任务
- 支持可配置的深度和特征通道数
"""

import torch
import torch.nn as nn


class ImageSharpnessUNet(nn.Module):
    """
    图像清晰度恢复的深度UNet模型
    
    特点：
    - 可配置的深度和特征通道数
    - 跳跃连接保留细节信息
    - 注意力机制增强特征选择
    - 适用于复杂的图像清晰度恢复任务
    """
    
    def __init__(self, in_channels=3, out_channels=3, features=None, depth=5, dropout=0.1, attention=True):
        super().__init__()
        
        # 默认特征配置
        if features is None:
            features = [64, 128, 256, 512, 1024]
        
        # 确保深度与特征数量匹配
        self.depth = min(depth, len(features))
        features = features[:self.depth]
        
        # 编码器
        self.encoder = nn.ModuleList()
        self.encoder.append(self._make_encoder_block(in_channels, features[0], dropout))
        
        for i in range(1, self.depth):
            self.encoder.append(self._make_encoder_block(features[i-1], features[i], dropout))
        
        # 解码器
        self.decoder = nn.ModuleList()
        for i in range(self.depth - 1):
            # 修复：计算跳跃连接后的总通道数
            # current_channels: 来自上采样后的特征通道数
            current_channels = features[self.depth-1-i]
            # skip_channels: 来自跳跃连接的特征通道数
            skip_channels = features[self.depth-2-i]
            # 总输入通道数 = current_channels + skip_channels
            total_input_channels = current_channels + skip_channels
            # 输出通道数
            output_channels = features[self.depth-2-i]
            
            self.decoder.append(self._make_decoder_block(total_input_channels, output_channels, dropout))
        
        # 最终输出层
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        # 池化和上采样
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # 注意力机制
        self.attention = attention
        if self.attention:
            self.attention_blocks = nn.ModuleList()
            # 修复：为每个跳跃连接创建对应通道数的注意力块
            for i in range(self.depth - 1):
                # 使用跳跃连接特征的通道数，而不是解码器特征的通道数
                skip_channels = features[self.depth-2-i]
                self.attention_blocks.append(AttentionBlock(skip_channels))
        
        # 批归一化
        self.batch_norms = nn.ModuleList()
        for i in range(self.depth):
            self.batch_norms.append(nn.BatchNorm2d(features[i]))
        
        # 激活函数
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        
    def _make_encoder_block(self, in_channels, out_channels, dropout):
        """创建编码器块"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def _make_decoder_block(self, in_channels, out_channels, dropout):
        """创建解码器块"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # 移除 *2，因为输入通道数已经包含跳跃连接
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        # 编码路径
        encoder_features = []
        current = x
        
        for i, encoder_block in enumerate(self.encoder):
            current = encoder_block(current)
            encoder_features.append(current)
            if i < self.depth - 1:  # 最后一层不进行池化
                current = self.pool(current)
        
        # 解码路径
        for i, decoder_block in enumerate(self.decoder):
            # 上采样
            current = self.upsample(current)
            
            # 跳跃连接
            skip_feature = encoder_features[self.depth-2-i]
            
            # 注意力机制
            if self.attention:
                skip_feature = self.attention_blocks[i](skip_feature)
            
            # 特征融合
            current = torch.cat([current, skip_feature], dim=1)
            current = decoder_block(current)
        
        # 最终输出
        output = self.final_conv(current)
        return output


class AttentionBlock(nn.Module):
    """注意力机制块，用于增强重要特征"""
    
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 空间注意力
        spatial_weights = self.spatial_attention(x)
        x_spatial = x * spatial_weights
        
        # 通道注意力
        channel_weights = self.channel_attention(x)
        x_channel = x * channel_weights
        
        # 组合注意力
        return x_spatial + x_channel


class ImageSharpnessUNetDeep(ImageSharpnessUNet):
    """深度UNet变体，使用更多特征通道"""
    
    def __init__(self, in_channels=3, out_channels=3):
        features = [64, 128, 256, 512, 1024, 2048]  # 增加特征通道
        super().__init__(in_channels, out_channels, features, depth=6, dropout=0.2, attention=True)


def build_image_sharpness_model(encoder_name: str = "unet", decoder_name: str = "unet", **kwargs):
    """
    构建图像清晰度模型
    
    Args:
        encoder_name: 编码器名称（支持unet, unet_deep）
        decoder_name: 解码器名称（支持unet, unet_deep）
        **kwargs: 其他参数
    
    Returns:
        构建的模型
    """
    if encoder_name.lower() == "unet" and decoder_name.lower() == "unet":
        return ImageSharpnessUNet(**kwargs)
    elif encoder_name.lower() == "unet_deep" and decoder_name.lower() == "unet_deep":
        return ImageSharpnessUNetDeep(**kwargs)
    else:
        # 如果指定了深度配置，使用自定义配置
        if "features" in kwargs or "depth" in kwargs:
            return ImageSharpnessUNet(**kwargs)
        else:
            raise ValueError(f"不支持的编码器/解码器组合: {encoder_name}/{decoder_name}")


# 便捷的模型构建函数
def create_lightweight_model(in_channels=3, out_channels=3):
    """创建轻量级模型（原始配置）"""
    return ImageSharpnessUNet(in_channels, out_channels, features=[32, 64, 128], depth=3, dropout=0.0, attention=False)


def create_standard_model(in_channels=3, out_channels=3):
    """创建标准模型（推荐配置）"""
    return ImageSharpnessUNet(in_channels, out_channels, features=[64, 128, 256, 512], depth=4, dropout=0.1, attention=True)


def create_deep_model(in_channels=3, out_channels=3):
    """创建深度模型（高容量配置）"""
    return ImageSharpnessUNet(in_channels, out_channels, features=[64, 128, 256, 512, 1024], depth=5, dropout=0.2, attention=True)


def create_ultra_deep_model(in_channels=3, out_channels=3):
    """创建超深度模型（最高容量配置）"""
    return ImageSharpnessUNet(in_channels, out_channels, features=[64, 128, 256, 512, 1024, 2048], depth=6, dropout=0.3, attention=True)
