# self_segmentation/configs/selfup/ssl_config.py
# 自监督学习 (SSL) - 掩码自编码器 (MAE) 风格的配置文件
from typing import Optional

# --- 数据集与路径配置 ---
UNLABELED_JSON_NAME: str = "master_unlabeled_dataset.json"

# --- 检查点路径配置 ---
# 训练完成后，最终的编码器权重保存路径
SSL_ENCODER_FINAL_PATH: str = "models/checkpoints/ssl_finetuned_encoder.pth"

# 用于保存损失最低的最佳编码器权重的文件名
SSL_BEST_MODEL_CHECKPOINT_PATH: str = "models/checkpoints/ssl_best_loss_encoder.pth"

# 用于断点续训的最新检查点文件名 (包含完整状态)
SSL_RESUMABLE_CHECKPOINT_PATH: str = "models/checkpoints/ssl_latest_checkpoint.pth"

# --- 恢复训练配置 ---
# 指向一个可恢复的检查点文件路径。设为 None 或 "" 则从头开始训练。
RESUME_FROM_CHECKPOINT: Optional[str] = "models/checkpoints/ssl_latest_checkpoint.pth"


# --- 训练超参数 ---
MODEL_NAME: str = "MAE_UNetEncoder"
BATCH_SIZE: int = 16
LEARNING_RATE: float = 5e-4
WEIGHT_DECAY: float = 0.05
NUM_EPOCHS: int = 10000    
NUM_WORKERS: int = 4

# --- 最佳模型保存频率 ---
SAVE_BEST_CHECK_EVERY_N_EPOCHS: int = 10


# --- MAE (掩码自编码器) 特定参数 ---
PATCH_SIZE: int = 16
MASK_RATIO: float = 0.08

# --- 解码器参数 ---
DECODER_EMBED_DIM: int = 256
DECODER_DEPTH: int = 4

# --- 日志记录与可视化 ---
LOG_DIR_NAME: str = "ssl_pretraining"
VISUALIZE_EVERY_N_EPOCHS: int = 10 
VISUALIZATION_DIR: str = "logs/ssl_visualizations"
