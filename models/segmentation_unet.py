from torch import nn
from .encoders.unet_encoder import UNetEncoder
from .decoders.unet_decoder import UNetDecoder

class SegmentationUNet(nn.Module):
    def __init__(self, encoder_name: str, decoder_name: str, n_channels: int = 3, n_classes: int = 3):
        super().__init__()

        if encoder_name != 'unet' or decoder_name != 'unet':
            raise ValueError("Currently, only 'unet' encoder and 'unet' decoder are supported.")

        self.encoder = UNetEncoder(n_channels=n_channels)

        # 自动检测encoder输出通道数，保证与自监督结构一致
        encoder_channels = self.encoder.get_channels()

        self.decoder = UNetDecoder(encoder_channels=encoder_channels, n_classes=n_classes)

    def forward(self, x):
        encoder_features = self.encoder(x)
        output = self.decoder(encoder_features)
        return output