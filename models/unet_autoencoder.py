import torch
import torch.nn as nn
from .encoders.unet_encoder import UNetEncoder
from .decoders.unet_decoder import UNetDecoder

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
            out = nn.functional.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return out 