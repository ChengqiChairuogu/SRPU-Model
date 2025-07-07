# self_segmentation/configs/finetune/train_config.py
from typing import Optional, Literal

# --- 任务与模型配置 ---
# 定义一个唯一的任务名称，用于创建日志和检查点目录
TASK_NAME: str = "sem_segmentation_baseline"

# 选择训练模式:
# 'from_scratch': 从随机初始化开始训练整个模型。
# 'finetune_frozen': 加载SSL预训练的编码器，并冻结它，只训练解码器。
# 'finetune_differential': 加载SSL预训练的编码器，并为编码器和解码器设置不同的学习率。
TRAINING_MODE: Literal['from_scratch', 'finetune_frozen', 'finetune_differential'] = 'finetune_differential'

# --- 路径配置 ---
# 用于微调的SSL预训练编码器权重的路径。
# 在 'finetune_*' 模式下，脚本会从这里加载权重。
# 这个路径应与 configs.selfup.ssl_config.SSL_BEST_MODEL_CHECKPOINT_PATH 或 SSL_ENCODER_FINAL_PATH 匹配。
PRETRAINED_ENCODER_PATH: Optional[str] = "models/checkpoints/ssl_best_loss_encoder.pth"

# 训练和验证使用的数据集JSON文件名。
# 这些文件应由 utils/json_generator.py 生成。
TRAIN_JSON_NAME: str = "master_labeled_dataset_train.json"
VAL_JSON_NAME: str = "master_labeled_dataset_val.json"

# --- 训练超参数 ---
BATCH_SIZE: int = 8
NUM_EPOCHS: int = 300
LOSS_FUNCTION: str = "DiceBCE" # 可选: "Focal", "Dice", "CrossEntropy" ... (需要在训练脚本中实现)

# --- 学习率配置 ---
# 基础学习率，用于 'from_scratch' 模式，或作为 'finetune_differential' 模式中解码器的学习率。
BASE_LEARNING_RATE: float = 1e-4

# 仅在 'finetune_differential' 模式下使用，用于预训练编码器的学习率。
# 通常设置得比基础学习率小一个数量级。
ENCODER_LEARNING_RATE: float = 1e-5

# 优化器配置
OPTIMIZER: str = "AdamW"
WEIGHT_DECAY: float = 1e-5

# --- 检查点与恢复训练 ---
# 是否只保存验证集指标最好的模型
SAVE_BEST_ONLY: bool = True
# 用于断点续训的检查点路径。设为 None 则从头开始。
RESUME_FROM_CHECKPOINT: Optional[str] = None # 例如: f"models/checkpoints/{TASK_NAME}/latest_checkpoint.pth"

# --- 日志记录 ---
# TensorBoard 日志将保存在 logs/TASK_NAME/ 下
