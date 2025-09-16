# self_segmentation/configs/finetune/finetune_config.py
from pathlib import Path
from configs import base as cfg_base

# --- 微调任务配置 ---
TASK_NAME = "sem_segmentation_finetuned_from_ssl"

# --- 模型架构选择 ---
ENCODER_NAME = "unet"
DECODER_NAME = "unet"

# --- 核心输入: 预训练编码器的路径 ---
# 注意：此路径应与 ssl_config.py 中的 BEST_MODEL_CHECKPOINT_PATH 匹配
PRETRAINED_ENCODER_PATH = Path(f"models/checkpoints/ssl_pretrained_{ENCODER_NAME}/best_ssl_encoder.pth")

# --- 数据集 ---
# 原始数据集
TRAIN_JSON_NAME = "master_labeled_dataset.json"
VAL_JSON_NAME = "master_labeled_dataset.json"

# 清晰度平均化后的数据集（推荐使用）
# TRAIN_JSON_NAME = "master_sharpness_averaged_dataset.json"
# VAL_JSON_NAME = "master_sharpness_averaged_dataset.json"

# 或者使用单个数据集的清晰度平均化JSON
# TRAIN_JSON_NAME = "dataset1_LInCl_sharpness_averaged.json"
# VAL_JSON_NAME = "dataset1_LInCl_sharpness_averaged.json"

# --- 微调策略 ---
# FINETUNE_MODE 可选值：
#   'finetune_frozen'：只训练解码器，编码器参数全部冻结（适合迁移学习初期或数据量少时）
#   'finetune_differential'：编码器和解码器都训练，但编码器使用较小学习率（推荐，适合大多数场景）
FINETUNE_MODE = 'finetune_differential'

# --- 训练超参数 ---
NUM_EPOCHS = 5000
BATCH_SIZE = 8
BASE_LEARNING_RATE = 1e-4
ENCODER_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-5
OPTIMIZER = "AdamW"
LOSS_FUNCTION = "DiceCELoss"
SEED = 42  # 全局随机种子，确保分层采样的一致性

# --- 断点续训配置 (统一逻辑) ---
RESUME_FROM_CHECKPOINT = True

# **关键修正**: 动态构建检查点目录，包含模型名称
CHECKPOINT_DIR_NAME = f"{TASK_NAME}_{ENCODER_NAME}_{DECODER_NAME}"
CHECKPOINT_DIR = Path(f"models/checkpoints/{CHECKPOINT_DIR_NAME}")
RESUMABLE_CHECKPOINT_PATH = CHECKPOINT_DIR / "resumable_checkpoint.pth"
BEST_MODEL_CHECKPOINT_PATH = CHECKPOINT_DIR / "best_model.pth"

# --- 日志记录配置 ---
LOGGER = "wandb"  # 可选 "wandb" 或 "tensorboard"
log_dir = cfg_base.BASE_LOG_DIR / LOGGER / TASK_NAME
log_config = {
    "logger": LOGGER,
    "project": "SRPU-Model",
    "log_dir": str(log_dir)
}