# configs/active_learning/active_learning_config.py
from pathlib import Path

# --- 主动学习任务配置 ---
TASK_NAME = "active_learning_sem_segmentation"

# --- 模型架构选择 ---
ENCODER_NAME = "unet"
DECODER_NAME = "unet"
ENCODER_WEIGHTS = None

# --- 预训练模型配置 ---
USE_PRETRAINED_MODEL = True
PRETRAINED_MODEL_PATH = Path("models/checkpoints/sem_segmentation_from_scratch_unet_unet/best_model.pth")  # 使用已训练的模型

# --- 数据集 ---
TRAIN_JSON_NAME = "master_labeled_dataset.json"
VAL_JSON_NAME = "master_labeled_dataset.json"

# --- 主动学习参数 ---
NUM_ITERATIONS = 20  # 迭代轮数（增加轮数，因为每次只标注一张）
SAMPLES_PER_ITER = 1  # 每轮选择的样本数（修改为1）

# 不确定性采样策略
# 传统方法: ["entropy", "margin", "least_confidence"]
# 边界采样方法: ["boundary_entropy", "boundary_margin", "gradient_uncertainty"]
# 混合方法: ["entropy", "boundary_entropy", "gradient_uncertainty"]
UNCERTAINTY_METHODS = ["boundary_entropy", "boundary_margin", "gradient_uncertainty"]

# 边界采样策略配置
BOUNDARY_SAMPLING_ENABLED = True
BOUNDARY_DILATION_SIZE = 3  # 边界膨胀像素数
MIN_BOUNDARY_AREA = 100     # 最小边界区域面积

# --- 训练参数（微调） ---
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
OPTIMIZER = "AdamW"
LOSS_FUNCTION = "DiceCELoss"
NUM_EPOCHS_PER_ITER = 10  # 每轮微调的epoch数

# --- 验证Dice系数的评估间隔 ---
DICE_EVAL_EPOCH_INTERVAL = 5  # 每轮微调中每隔5个epoch评估一次

# --- 断点续训配置 ---
RESUME_FROM_CHECKPOINT = True

# --- 检查点路径（与训练一致的组织方式） ---
CHECKPOINT_DIR_NAME = f"{TASK_NAME}_{ENCODER_NAME}_{DECODER_NAME}"
CHECKPOINT_DIR = Path(f"models/checkpoints/{CHECKPOINT_DIR_NAME}")
RESUMABLE_CHECKPOINT_PATH = CHECKPOINT_DIR / "resumable_checkpoint.pth"
BEST_MODEL_CHECKPOINT_PATH = CHECKPOINT_DIR / "best_model.pth"

# --- 数据路径 ---
UNLABELED_POOL_DIR = Path("data/raw/")
LABELED_DIR = Path("data/labeled/")
PREDICTIONS_DIR = Path("active_learning/predictions/")
UNCERTAINTY_MAPS_DIR = Path("active_learning/uncertainty_maps/")
SELECTION_INFO_DIR = Path("active_learning/selection_info/")

# --- 日志记录配置 ---
LOGGER = "tensorboard"  # 默认使用tensorboard，可选 "wandb" 或 "tensorboard"
log_dir = f"runs/{LOGGER}/{TASK_NAME}"
log_config = {
    "logger": LOGGER,
    "project": "SRPU-Model",
    "log_dir": log_dir
} 