# 文件路径: models/encoders/dinov2_encoder.py

import torch
import torch.nn as nn
import timm

ENCODER_CHANNELS = {
    # DINOv2 ViT-L/14, 这是一个Transformer模型，特征层级和CNN不同
    # 通常取其块(block)的输出作为特征
    "dinov2_vitl14": [1024] * 4 # 示例：假设我们从4个等距的块中提取特征
}

class DinoV2Encoder(nn.Module):
    def __init__(self, encoder_name='vit_large_patch14_dinov2.lvd142m', pretrained=True, in_channels=3):
        """
        初始化 DINOv2 编码器。
        
        参数:
        - encoder_name (str): timm中DINOv2模型的名称。
        - pretrained (bool): 是否加载预训练权重。
        - in_channels (int): 输入通道数。
        """
        super().__init__()
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            in_chans=in_channels,
            features_only=True, # 让模型返回中间层的特征
            out_indices=(5, 11, 17, 23) # ViT-L有24个块，我们取4个作为特征层
        )
        self.channels = self.encoder.feature_info.channels()

    def get_channels(self):
        # 返回timm提供的各特征层通道数
        return self.channels

    def forward(self, x):
        # timm模型直接返回一个特征列表
        return self.encoder(x)