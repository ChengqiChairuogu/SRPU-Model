from torch import nn
from .encoders.unet_encoder import UNetEncoder
from .decoders.unet_decoder import UNetDecoder

class SegmentationUNet(nn.Module):
    def __init__(self, encoder=None, decoder=None, encoder_name: str = None, decoder_name: str = None, n_channels: int = 3, n_classes: int = 3):
        super().__init__()
        # 兼容原有接口，也支持直接传入encoder/decoder实例
        if encoder is not None and decoder is not None:
            self.encoder = encoder
            self.decoder = decoder
        else:
            if encoder_name != 'unet' or decoder_name != 'unet':
                raise ValueError("Currently, only 'unet' encoder and 'unet' decoder are supported.")
            self.encoder = UNetEncoder(n_channels=n_channels)
            encoder_channels = self.encoder.get_channels()
            self.decoder = UNetDecoder(encoder_channels=encoder_channels, n_classes=n_classes)

    def forward(self, x):
        encoder_features = self.encoder(x)
        output = self.decoder(encoder_features)
        return output