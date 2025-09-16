# self_segmentation/configs/train/train_config.py
from pathlib import Path
from configs import base as cfg_base

# --- 监督学习任务配置 (基线实验) ---
TASK_NAME = "sem_segmentation_from_scratch"

# --- 模型架构选择 ---
ENCODER_NAME = "unet"
DECODER_NAME = "unet"
ENCODER_WEIGHTS = None

# --- 数据集 ---
TRAIN_JSON_NAME = "master_labeled_dataset.json"
VAL_JSON_NAME = "master_labeled_dataset.json"

# --- 训练超参数 ---
NUM_EPOCHS = 5000
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
OPTIMIZER = "AdamW"
LOSS_FUNCTION = "DiceCELoss"

# --- 验证Dice系数的评估间隔 ---
DICE_EVAL_EPOCH_INTERVAL = 10  # 每隔10个epoch评估一次val各类别dice系数

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