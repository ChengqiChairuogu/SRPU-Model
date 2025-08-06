from pathlib import Path
from configs import base as cfg_base

# --- 多阶段微调任务配置 ---

STAGES = [
    {
        "name": "stage1",
        "train_json": "master_labeled_dataset.json",  # 可自定义json
        "val_json": "master_labeled_dataset.json",
        "datasets": ["dataset1_LInCl", "dataset2_LPSCl"],  # 参与本阶段的子数据集名
        "num_epochs": 20,
        "batch_size": 8,
        "base_lr": 1e-4,
        "encoder_lr": 1e-5,
        "finetune_mode": "finetune_differential",
        "pretrained_encoder_path": Path("models/checkpoints/ssl_pretrained_unet/best_ssl_encoder.pth"),  # 第一阶段可选
        "pretrained_model_path": None,  # 第一阶段可指定完整预训练模型路径
    },
    {
        "name": "stage2",
        "train_json": "master_labeled_dataset.json",
        "val_json": "master_labeled_dataset.json",
        "datasets": ["dataset3_LNOCl"],
        "num_epochs": 10,
        "batch_size": 8,
        "base_lr": 5e-5,
        "encoder_lr": 1e-6,
        "finetune_mode": "finetune_differential",
        # pretrained_encoder_path留空，自动用stage1输出
        "pretrained_model_path": None,  # 第二阶段自动使用前一阶段模型
    }
]

# --- 预训练模型配置 ---
USE_PRETRAINED_MODEL = False  # 是否使用预训练模型
PRETRAINED_MODEL_PATH = Path("models/checkpoints/your_pretrained_model.pth")  # 预训练模型路径

# --- 模型权重保存配置 ---
MODEL_SAVE_DIR = Path("models/checkpoints/multistage_finetune")  # 模型权重保存目录
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
log_dir = cfg_base.BASE_LOG_DIR / LOGGER / f"multistage_finetune_{ENCODER_NAME}_{DECODER_NAME}"
log_config = {
    "logger": LOGGER,
    "project": "SRPU-Model",
    "log_dir": str(log_dir)
} 