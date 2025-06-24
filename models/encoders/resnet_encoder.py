# 文件路径: models/encoders/resnet_encoder.py

import torch.nn as nn
import torchvision.models as models

# 为不同的ResNet变体预定义输出通道数
# 格式: [stage0, stage1, stage2, stage3, stage4] channels
ENCODER_CHANNELS = {
    "resnet34": [64, 64, 128, 256, 512],
    "resnet50": [64, 256, 512, 1024, 2048],
}

class ResNetEncoder(nn.Module):
    def __init__(self, encoder_name='resnet50', pretrained=True, in_channels=3):
        """
        初始化ResNet编码器。
        
        参数:
        - encoder_name (str): 'resnet34' 或 'resnet50'.
        - pretrained (bool): 是否加载ImageNet预训练权重。
        - in_channels (int): 输入图像通道数。
        """
        super().__init__()
        
        if encoder_name not in ENCODER_CHANNELS:
            raise ValueError(f"不支持的 Encoder: {encoder_name}")
            
        self.encoder_name = encoder_name

        if encoder_name == 'resnet34':
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            self.base_model = models.resnet34(weights=weights)
        elif encoder_name == 'resnet50':
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            self.base_model = models.resnet50(weights=weights)

        if in_channels != 3:
            original_conv1 = self.base_model.conv1
            new_conv1 = nn.Conv2d(in_channels, original_conv1.out_channels, 
                                  kernel_size=original_conv1.kernel_size, 
                                  stride=original_conv1.stride, 
                                  padding=original_conv1.padding, 
                                  bias=original_conv1.bias)
            if pretrained:
                original_weights = original_conv1.weight.data
                mean_weights = original_weights.mean(dim=1, keepdim=True)
                new_weights = mean_weights.repeat(1, in_channels, 1, 1)
                new_conv1.weight.data = new_weights
            self.base_model.conv1 = new_conv1

        self.base_model.fc = nn.Identity()
        self.base_model.avgpool = nn.Identity()

    def get_channels(self):
        return ENCODER_CHANNELS[self.encoder_name]

    def forward(self, x):
        features = []
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        features.append(x)
        
        x = self.base_model.maxpool(x)
        x = self.base_model.layer1(x)
        features.append(x)
        
        x = self.base_model.layer2(x)
        features.append(x)
        
        x = self.base_model.layer3(x)
        features.append(x)
        
        x = self.base_model.layer4(x)
        features.append(x)
        
        return features