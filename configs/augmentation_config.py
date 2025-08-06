# self_segmentation/configs/augmentation_config.py
# 在这里配置训练和验证时使用的数据增强策略

# --- 通用设置 ---
# 裁剪的目标尺寸，将从 base.py 中自动读取，无需在此设置
# CROP_HEIGHT = 512
# CROP_WIDTH = 512

# ==============================================================================
#                      训练集 (TRAIN) 数据增强配置
# ==============================================================================
# 将 'enabled' 设置为 True 或 False 来开启或关闭对应的增强功能

TRAIN_AUGMENTATIONS = {
    # --- 裁剪 ---
    "random_crop": {
        "enabled": True,  # 始终对训练集启用随机裁剪以匹配目标尺寸
    },

    # --- 几何变换 ---
    "horizontal_flip": {
        "enabled": True,
        "p": 0.5,  # 应用此变换的概率
    },
    "vertical_flip": {
        "enabled": True,
        "p": 0.5,
    },
    "random_rotate_90": {
        "enabled": True,
        "p": 0.5,
    },
    "rotate": {
        "enabled": False, # 更自由的旋转，可能会引入黑色边框，慎用
        "p": 0.5,
        "limit": (-30, 30), # 旋转角度范围
        "border_mode": "constant", # 填充边框的模式
        "value": 0, # 填充值
    },

    # --- 颜色和噪声 ---
    "random_brightness_contrast": {
        "enabled": False,
        "p": 0.5,
        "brightness_limit": 0.2,
        "contrast_limit": 0.2,
    },
    "gaussian_blur": {
        "enabled": False,
        "p": 0.3,
        "blur_limit": (3, 7), # 模糊核的大小范围
    },
    "gauss_noise": {
        "enabled": False,
        "p": 0.3,
        "var_limit": (10.0, 50.0), # 噪声方差范围
    },
}


# ==============================================================================
#                   验证集/测试集 (VAL/TEST) 数据增强配置
# ==============================================================================
# 通常验证集只做必要的尺寸匹配（如中心裁剪），不做随机性增强

VAL_AUGMENTATIONS = {
    "center_crop": {
        "enabled": True, # 始终对验证集启用中心裁剪以匹配目标尺寸
    }
}


# ==============================================================================
#                    自监督学习 (SSL) 数据增强配置
# ==============================================================================
# SSL通常使用与监督学习相似但可能更强的几何增强

SSL_AUGMENTATIONS = {
    "random_crop": {
        "enabled": True,
    },
    "horizontal_flip": {
        "enabled": True,
        "p": 0.5,
    },
    "vertical_flip": {
        "enabled": True,
        "p": 0.5,
    },
    "random_rotate_90": {
        "enabled": True,
        "p": 0.5,
    },
}