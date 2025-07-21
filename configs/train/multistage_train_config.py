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
    },
    {
        "name": "stage2",
        "train_json": "master_labeled_dataset.json",
        "val_json": "master_labeled_dataset.json",
        "datasets": ["dataset3_LNOCl"],
        "num_epochs": 10,
        "batch_size": 8,
        "base_lr": 5e-5,
    }
]

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