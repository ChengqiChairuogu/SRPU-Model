# self_segmentation/configs/finetune/finetune_config.py
from pathlib import Path

# --- 微调任务配置 ---
TASK_NAME = "sem_segmentation_finetuned_from_ssl"

# --- 模型架构选择 ---
ENCODER_NAME = "unet"
DECODER_NAME = "unet"

# --- 核心输入: 预训练编码器的路径 ---
# 注意：此路径应与 ssl_config.py 中的 BEST_MODEL_CHECKPOINT_PATH 匹配
PRETRAINED_ENCODER_PATH = Path(f"models/checkpoints/ssl_pretrained_{ENCODER_NAME}/best_ssl_encoder.pth")

# --- 数据集 ---
TRAIN_JSON_NAME = "master_labeled_dataset_train.json"
VAL_JSON_NAME = "master_labeled_dataset_val.json"

# --- 微调策略 ---
FINETUNE_MODE = 'finetune_differential'

# --- 训练超参数 ---
NUM_EPOCHS = 5000
BATCH_SIZE = 8
BASE_LEARNING_RATE = 1e-4
ENCODER_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-5
OPTIMIZER = "AdamW"
LOSS_FUNCTION = "DiceBCELoss"

# --- 断点续训配置 (统一逻辑) ---
RESUME_FROM_CHECKPOINT = True

# **关键修正**: 动态构建检查点目录，包含模型名称
CHECKPOINT_DIR_NAME = f"{TASK_NAME}_{ENCODER_NAME}_{DECODER_NAME}"
CHECKPOINT_DIR = Path(f"models/checkpoints/{CHECKPOINT_DIR_NAME}")
RESUMABLE_CHECKPOINT_PATH = CHECKPOINT_DIR / "resumable_checkpoint.pth"
BEST_MODEL_CHECKPOINT_PATH = CHECKPOINT_DIR / "best_model.pth"