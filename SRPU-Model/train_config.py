# self_segmentation/configs/train/train_config.py
# 用于“从零开始”训练的配置文件

# --- 任务与模型配置 ---
# 定义一个唯一的任务名称，用于创建日志和检查点目录
TASK_NAME: str = "sem_segmentation_from_scratch"

# --- 数据集配置 ---
# 训练和验证使用的数据集JSON文件名。
# 这些文件应由 utils/json_generator.py 生成。
TRAIN_JSON_NAME: str = "train_labeled_dataset.json"
VAL_JSON_NAME: str = "val_labeled_dataset.json"

# --- 训练超参数 ---
BATCH_SIZE: int = 8
NUM_EPOCHS: int = 300
# 基础学习率，用于训练整个模型
LEARNING_RATE: float = 1e-4

# --- 优化器与损失函数 ---
OPTIMIZER: str = "AdamW"
WEIGHT_DECAY: float = 1e-5
LOSS_FUNCTION: str = "DiceBCE" # 可选: "Focal", "Dice", "CrossEntropy"

# --- 检查点与恢复训练 ---
# 是否只保存验证集指标最好的模型
SAVE_BEST_ONLY: bool = True
# 用于断点续训的检查点路径。设为 None 则从头开始。
RESUME_FROM_CHECKPOINT: str = "" # 例如: f"models/checkpoints/{TASK_NAME}/latest_checkpoint.pth"
