# 文件路径: models/decoders/unet_decoder.py

import torch
import torch.nn as nn

class DecoderBlock(nn.Module):
    def __init__(self, in_channels_skip, in_channels_prev, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels_prev, in_channels_prev // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels_skip + in_channels_prev // 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x_skip, x_prev):
        x_prev = self.upsample(x_prev)
        x = torch.cat([x_skip, x_prev], dim=1)
        return self.conv(x)

class UNetDecoder(nn.Module):
    def __init__(self, encoder_channels, n_classes):
        """
        参数:
        - encoder_channels (list): 编码器输出的各阶段特征图的通道数，从浅到深。
                                   例如 ResNet50: [64, 256, 512, 1024, 2048]。
        """
        super().__init__()
        reversed_encoder_channels = list(reversed(encoder_channels))
        # 新的解码通道数，适配6层更深更宽
        decoder_channels = [2048, 1024, 512, 256, 128]
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(reversed_encoder_channels[1], reversed_encoder_channels[0], decoder_channels[0])
        ])
        for i in range(1, len(decoder_channels)):
            self.decoder_blocks.append(
                DecoderBlock(reversed_encoder_channels[i+1], decoder_channels[i-1], decoder_channels[i])
            )
        self.final_upsample = nn.ConvTranspose2d(decoder_channels[-1], decoder_channels[-1] // 2, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(decoder_channels[-1] // 2, n_classes, kernel_size=1)

    def forward(self, features):
        features = list(reversed(features))
        x = features[0]
        for i, decoder_block in enumerate(self.decoder_blocks):
            skip_connection = features[i + 1]
            x = decoder_block(skip_connection, x)
        x = self.final_upsample(x)
        x = self.final_conv(x)
        return x