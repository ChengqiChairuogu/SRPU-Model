from pathlib import Path

# --- 1. 项目路径配置 (Project Path Configuration) ---
# 项目根目录，Pathlib使得路径操作更简单、跨平台
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 数据相关路径
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"          # 存放所有原始 .tif 图像 
LABELED_DATA_DIR = DATA_DIR / "labeled"    # 存放已标注的图像和掩码 
JSON_DIR = PROJECT_ROOT / "json"             # 存放数据集索引的json文件
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints" # 存放模型权重

# --- 2. 数据集通用配置 (General Dataset Configuration) ---
# 图像基本属性
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512

# 决定模型输入通道数，对应你U-Net中的 n_channels 
INPUT_DEPTH = 3

# --- 3. 模型与类别配置 (Model and Class Configuration) ---
# 分割的类别数量，对应你U-Net中的 n_classes 
NUM_CLASSES = 3  # 只分AM、SE、carbon三类

# U-Net解码器上采样时是否使用双线性插值 
BILINEAR = True

# 颜色到类别的映射 (RGB -> 类别ID)
MAPPING = {
    (0,   0, 255): 0,   # carbon
    (0, 255, 255): 1,   # SE
    (255,126,126): 2    # AM
}

# 类别ID到颜色的映射 (类别ID -> RGB) - 与MAPPING保持一致
COLOR_MAPPING = {
    0: [0, 0, 255],      # carbon
    1: [0, 255, 255],    # SE - 青色
    2: [255, 126, 126]   # AM - 红色
}

# 类别名称映射
CLASS_NAMES = {
    0: "carbon", 
    1: "SE", 
    2: "AM"
}

NUM_CLASSES = 3

# --- 4. JSON文件命名 (JSON Filename Configuration) ---
TRAIN_JSON_FILENAME = "train_dataset.json"
VALID_JSON_FILENAME = "valid_dataset.json"
TEST_JSON_FILENAME = "test_dataset.json"
MASTER_UNLABELED_JSON_FILENAME = "master_unlabeled_dataset.json"
MASTER_LABELED_JSON_FILENAME = "master_labeled_dataset.json"

# --- 5. 工具与日志配置 (Tools and Logging Configuration) ---
WANDB_ENABLED = True  # 设置为 True 启用, 设置为 False 禁用