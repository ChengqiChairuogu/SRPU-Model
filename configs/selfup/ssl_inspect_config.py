from pathlib import Path

# --- 自监督评估任务配置 ---
EVAL_NUM_IMAGES = 1  # 评估时随机选取的图片数量
MASK_RATIO = 0.5   # 挖去区域的比例
BATCH_SIZE = 1
JSON_DIR_NAME = "json"
UNLABELED_JSON_NAME = "master_ssl_dataset.json"
MODEL_CHECKPOINT_PATH = Path("models/checkpoints/ssl_pretrained_unet/best_ssl_encoder.pth")
OUTPUT_DIR = Path("data/ssl_inspect_results/") 
MODEL_TYPE = 'unet_autoencoder'  # 可选: 'unet_autoencoder', 'mae_unet' 
DEBUG_MODE = True  # 是否打印权重加载和参数均值等详细log 
COMPARE_INDEX = 290  # 用于训练-评估对比的图片索引（如设为None则随机） 
SEED = 42  # 全局随机种子，保证训练和inspect一致 