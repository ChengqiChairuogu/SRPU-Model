# self_segmentation/configs/train/train_config.py
from pathlib import Path

# --- 监督学习任务配置 (基线实验) ---
TASK_NAME = "sem_segmentation_from_scratch"

# --- 模型架构选择 ---
ENCODER_NAME = "unet"
DECODER_NAME = "unet"
ENCODER_WEIGHTS = None

# --- 数据集 ---
TRAIN_JSON_NAME = "master_labeled_dataset_train.json"
VAL_JSON_NAME = "master_labeled_dataset_val.json"

# --- 训练超参数 ---
NUM_EPOCHS = 5000
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
OPTIMIZER = "AdamW"
LOSS_FUNCTION = "DiceBCELoss"

# --- 断点续训配置 (统一逻辑) ---
RESUME_FROM_CHECKPOINT = False

# --- 动态构建检查点目录，包含模型名称 ---
CHECKPOINT_DIR_NAME = f"{TASK_NAME}_{ENCODER_NAME}_{DECODER_NAME}"
CHECKPOINT_DIR = Path(f"models/checkpoints/{CHECKPOINT_DIR_NAME}")
RESUMABLE_CHECKPOINT_PATH = CHECKPOINT_DIR / "resumable_checkpoint.pth"
BEST_MODEL_CHECKPOINT_PATH = CHECKPOINT_DIR / "best_model.pth"