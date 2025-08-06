# self_segmentation/configs/selfup/ssl_config.py
from pathlib import Path

# --- 自监督学习 (SSL) 配置 ---
# --- 编码器架构选择 ---
ENCODER_NAME = "unet"
MODEL_NAME = f"mae_{ENCODER_NAME}"

# --- 数据集 ---
JSON_DIR_NAME = "json"
UNLABELED_JSON_NAME = "master_ssl_dataset.json"

# --- 训练超参数 ---
NUM_EPOCHS = 5000
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 4

# --- MAE 特定参数 ---
PATCH_SIZE = 16
MASK_RATIO = 0.5
DECODER_EMBED_DIM = 512
DECODER_DEPTH = 4

# --- 断点续训配置 (统一逻辑) ---
RESUME_FROM_CHECKPOINT = False

# **关键修正**: 动态构建检查点目录
CHECKPOINT_DIR_NAME = f"ssl_pretrained_{ENCODER_NAME}"
SSL_CHECKPOINT_DIR = Path(f"models/checkpoints/{CHECKPOINT_DIR_NAME}")
RESUMABLE_CHECKPOINT_PATH = SSL_CHECKPOINT_DIR / "resumable_checkpoint.pth"
BEST_MODEL_CHECKPOINT_PATH = SSL_CHECKPOINT_DIR / "best_ssl_encoder.pth"
SSL_ENCODER_FINAL_PATH = SSL_CHECKPOINT_DIR / "ssl_encoder_final.pth"

SAVE_BEST_CHECK_EVERY_N_EPOCHS = 10
SEED = 42  # 全局随机种子，保证训练和inspect一致

# --- 日志记录配置 ---
LOGGER = 'tensorboard'
log_dir = f'runs/tensorboard/{MODEL_NAME}'
log_config = {
    'logger': LOGGER,
    'project': 'SRPU-Model',
    'log_dir': log_dir
}
EVAL_INTERVAL = 10  # 评估指标计算间隔（epoch数）