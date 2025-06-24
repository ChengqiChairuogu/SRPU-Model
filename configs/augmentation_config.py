# --- 归一化时使用的像素值范围最大值 ---
MAX_PIXEL_VALUE = 255.0 

# --- 目标图像尺寸 (用于Resize或Crop) ---
from .base import IMAGE_HEIGHT,IMAGE_WIDTH
TARGET_HEIGHT = IMAGE_HEIGHT
TARGET_WIDTH = IMAGE_WIDTH

# --- 训练集数据增强参数 ---
TRAIN_AUGMENTATIONS_PARAMS = {
    "RandomResizedCrop": {
        "height": TARGET_HEIGHT, "width": TARGET_WIDTH,
        "scale": (0.7, 1.0), "ratio": (0.75, 1.33),
        "interpolation": 1, "p": 0.6
    },
    "ResizeWhenNeeded": { # 这是一个逻辑名称，我们会在build_augmentations中处理
        "height": TARGET_HEIGHT, "width": TARGET_WIDTH, "interpolation": 1
    },
    "HorizontalFlip": {"p": 0.5},
    "VerticalFlip": {"p": 0.5},
    "RandomRotate90": {"p": 0.5},
    "ShiftScaleRotate": {
        "p": 0.4, "shift_limit": 0.07, "scale_limit": 0.15,
        "rotate_limit": 35, "interpolation": 1,
        "border_mode": 0, "value": 0, "mask_value": 0
    },
    "OneOf_Color_Pixel": {
        "p": 0.5,
        "transforms": [
            {"type": "RandomBrightnessContrast", "brightness_limit":0.25, "contrast_limit":0.25, "p":1.0},
            {"type": "HueSaturationValue", "hue_shift_limit":10, "sat_shift_limit":15, "val_shift_limit":10, "p":1.0}
        ]
    },
    "GaussNoise": {"var_limit": (10.0, 50.0), "p": 0.2},
    # Normalize 和 ToTensorV2 将由 build_augmentations 自动添加
}

# --- 验证/测试集数据转换参数 ---
VAL_TRANSFORMS_PARAMS = {
    "Resize": {
        "height": TARGET_HEIGHT, "width": TARGET_WIDTH,
        "interpolation": 1, "always_apply": True
    },
    # Normalize 和 ToTensorV2 将由 build_augmentations 自动添加
}
