#!/usr/bin/env python3
# pretrained_model_config.py
"""
预训练模型管理配置文件

定义预训练模型管理工具的各种配置参数
"""

# --- 基本操作配置 ---
# 操作类型: list(列出模型), info(显示模型信息), validate(验证模型), summary(创建摘要)
ACTION = "list"

# 指定模型文件名（用于info和validate操作）
MODEL_NAME = None

# 模型目录路径
MODELS_DIR = "models/checkpoints"

# --- 自动化配置 ---
# 是否在列出模型后自动验证所有模型
AUTO_VALIDATE = False

# 是否在操作完成后自动创建模型摘要
CREATE_SUMMARY = False

# --- 输出配置 ---
# 输出格式: text(文本), json(JSON格式), csv(CSV格式)
OUTPUT_FORMAT = "text"

# --- 验证配置 ---
# 验证时使用的设备类型: cpu, cuda, auto
VALIDATION_DEVICE = "cpu"

# 是否在验证失败时继续验证其他模型
CONTINUE_ON_VALIDATION_ERROR = True

# --- 摘要配置 ---
# 摘要输出文件路径
SUMMARY_OUTPUT_PATH = "model_summary.txt"

# 是否包含模型详细信息
INCLUDE_DETAILED_INFO = True

# 是否按性能排序
SORT_BY_PERFORMANCE = True
