from pathlib import Path

# --- 多阶段有监督训练任务配置 ---

STAGES = [
    {
        "name": "stage1",
        "train_json": "master_labeled_dataset.json",
        "val_json": "master_labeled_dataset.json",
        "datasets": ["dataset1_LInCl", "dataset2_LPSCl"],
        "num_epochs": 20,
        "batch_size": 8,
        "base_lr": 1e-4,
        "pretrained_model_path": None,  # 第一阶段可指定预训练模型路径
    },
    {
        "name": "stage2",
        "train_json": "master_labeled_dataset.json",
        "val_json": "master_labeled_dataset.json",
        "datasets": ["dataset3_LNOCl"],
        "num_epochs": 10,
        "batch_size": 8,
        "base_lr": 5e-5,
        "pretrained_model_path": None,  # 第二阶段自动使用前一阶段模型
    }
]

# --- 预训练模型配置 ---
USE_PRETRAINED_MODEL = False  # 是否使用预训练模型
PRETRAINED_MODEL_PATH = Path("models/checkpoints/your_pretrained_model.pth")  # 预训练模型路径

# --- 模型权重保存配置 ---
MODEL_SAVE_DIR = Path("models/checkpoints/multistage_train")  # 模型权重保存目录
SAVE_BEST_MODEL = True  # 是否保存最佳模型
SAVE_LAST_MODEL = False  # 是否保存最后一个epoch的模型
SAVE_CHECKPOINT = True  # 是否保存训练检查点（包含优化器状态）

# --- 模型结构与训练通用参数 ---
ENCODER_NAME = "unet"
DECODER_NAME = "unet"
LOSS_FUNCTION = "DiceCELoss"
OPTIMIZER = "AdamW"
WEIGHT_DECAY = 1e-5
SEED = 42
DICE_EVAL_EPOCH_INTERVAL = 10  # 每隔10个epoch输出一次详细dice

# --- 日志配置 ---
LOGGER = "wandb"  # 可选 "wandb" 或 "tensorboard"
log_dir = f"runs/{LOGGER}/multistage_train_{ENCODER_NAME}_{DECODER_NAME}"
log_config = {
    "logger": LOGGER,
    "project": "SRPU-Model",
    "log_dir": log_dir
} 