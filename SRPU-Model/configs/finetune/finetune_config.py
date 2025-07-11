# self_segmentation/configs/finetune/finetune_config.py
# 用于“微调”预训练模型的配置文件
from typing import Optional, Literal

# --- 任务与模型配置 ---
TASK_NAME: str = "sem_segmentation_finetune_ssl"

# --- 微调模式 ---
# 'finetune_frozen': 加载SSL预训练的编码器，并冻结它，只训练解码器。
# 'finetune_differential': 加载SSL预训练的编码器，并为编码器和解码器设置不同的学习率。
FINETUNE_MODE: Literal['finetune_frozen', 'finetune_differential'] = 'finetune_differential'

# --- 路径配置 ---
# 用于微调的SSL预训练编码器权重的路径。
# 在 'finetune_*' 模式下，脚本会从这里加载权重。
# 这个路径应与 configs.selfup.ssl_config 中的检查点路径匹配。
PRETRAINED_ENCODER_PATH: Optional[str] = "models/checkpoints/ssl_best_loss_encoder.pth"

# --- 数据集配置 ---
# 训练和验证使用的数据集JSON文件名。
# 这些文件名现在与您的 json_generator.py 输出匹配。
TRAIN_JSON_NAME: str = "master_labeled_dataset_train.json"
VAL_JSON_NAME: str = "master_labeled_dataset_val.json"

# --- 训练超参数 ---
BATCH_SIZE: int = 8
NUM_EPOCHS: int = 200

# --- 学习率配置 ---
# 基础学习率，用于 'finetune_frozen' 模式中的解码器，
# 或 'finetune_differential' 模式中解码器的学习率。
BASE_LEARNING_RATE: float = 1e-4

# 仅在 'finetune_differential' 模式下使用，用于预训练编码器的学习率。
# 通常设置得比基础学习率小一个数量级。
ENCODER_LEARNING_RATE: float = 1e-5

# --- 优化器与损失函数 ---
OPTIMIZER: str = "AdamW"
WEIGHT_DECAY: float = 1e-5
LOSS_FUNCTION: str = "DiceBCE"

# --- 检查点与恢复训练 ---
SAVE_BEST_ONLY: bool = True
RESUME_FROM_CHECKPOINT: Optional[str] = None
