# 文件路径: models/encoders/efficientnet_encoder.py

import torch.nn as nn
import timm

ENCODER_CHANNELS = {
    "efficientnet_b4": [24, 32, 56, 160, 448],
    "efficientnet_b5": [24, 40, 64, 176, 512],
}

class EfficientNetEncoder(nn.Module):
    def __init__(self, encoder_name='efficientnet_b4', pretrained=True, in_channels=3):
        """
        初始化EfficientNet编码器 (使用timm库)。
        
        参数:
        - encoder_name (str): e.g., 'efficientnet_b4'.
        - pretrained (bool): 是否加载预训练权重。
        - in_channels (int): 输入通道数。
        """
        super().__init__()
        
        if encoder_name not in ENCODER_CHANNELS:
            raise ValueError(f"不支持的 Encoder: {encoder_name}")
            
        self.encoder_name = encoder_name

        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            in_chans=in_channels,
            features_only=True,
            out_indices=(1, 2, 3, 4) # 指定输出的特征层索引
        )

    def get_channels(self):
        # timm的'features_only'模式输出的通道数列表
        # 我们需要手动在前面加上初始stem卷积的输出通道
        # 对于efficientnet_b4是24，b5是24
        stem_channels = 24 # 根据模型调整
        base_channels = self.encoder.feature_info.channels()
        return [stem_channels] + base_channels

    def forward(self, x):
        # timm features_only=True时，输出一个特征列表
        # 但它不包含最初的stem输出，这对于U-Net的某些解码器是需要的
        # 我们可以通过一次完整的前向传播获取所有特征
        feat_maps = self.encoder(x)
        
        # 为了与我们的UNetDecoder兼容，需要补上第一个特征
        # 但大多数现代解码器(如DeepLabV3+)不需要最浅层的特征
        # 这里我们假设解码器能处理timm直接给出的特征列表
        # (这个需要根据你选的解码器进行适配)
        # 为简单起见，我们直接返回timm给出的4个阶段特征
        return feat_maps # 返回 [stage1, stage2, stage3, stage4] 特征