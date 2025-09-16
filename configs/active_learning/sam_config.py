# configs/active_learning/sam_config.py
from pathlib import Path

# --- SAM模型配置 ---
SAM_ENABLED = True
SAM_CHECKPOINT_PATH = "models/sam/sam_vit_h_4b8939.pth"  # SAM模型权重路径
SAM_MODEL_TYPE = "vit_h"  # 模型类型: "vit_h", "vit_l", "vit_b"

# --- SAM提示词配置 ---
SAM_PROMPT_TYPES = {
    "click": {
        "enabled": True,
        "description": "点击前景/背景点进行分割"
    },
    "box": {
        "enabled": True,
        "description": "绘制边界框进行分割"
    },
    "text": {
        "enabled": False,  # 需要CLIP集成
        "description": "文本描述分割"
    }
}

# --- 交互式标注配置 ---
INTERACTIVE_ANNOTATION = {
    "enabled": True,
    "interface_type": "matplotlib",  # "matplotlib", "tkinter", "web"
    "auto_save": True,
    "save_format": "png"
}

# --- 修正策略配置 ---
CORRECTION_STRATEGIES = {
    "automatic_refinement": True,  # 自动细化边界
    "multi_mask_selection": True,  # 多mask选择
    "confidence_threshold": 0.8,   # 置信度阈值
    "iou_threshold": 0.5          # IoU阈值
}

# --- 类别特定提示词 ---
CLASS_PROMPTS = {
    0: {  # carbon
        "keywords": ["carbon", "black", "dark", "particle"],
        "visual_features": ["圆形", "颗粒状", "深色"]
    },
    1: {  # SE
        "keywords": ["SE", "secondary", "electron", "bright"],
        "visual_features": ["明亮", "边缘", "表面"]
    },
    2: {  # AM
        "keywords": ["AM", "active", "material", "crystal"],
        "visual_features": ["晶体", "规则形状", "高对比度"]
    }
}

# --- 标注效率优化 ---
ANNOTATION_EFFICIENCY = {
    "batch_processing": True,      # 批量处理
    "smart_suggestions": True,     # 智能建议
    "shortcut_keys": True,         # 快捷键支持
    "auto_completion": True,       # 自动补全
    "template_matching": True      # 模板匹配
}

# --- 质量控制 ---
QUALITY_CONTROL = {
    "consistency_check": True,     # 一致性检查
    "boundary_refinement": True,   # 边界细化
    "overlap_detection": True,     # 重叠检测
    "size_validation": True        # 尺寸验证
} 