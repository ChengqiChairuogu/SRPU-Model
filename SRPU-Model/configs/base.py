from pathlib import Path

# --- 1. 项目路径配置 (Project Path Configuration) ---
# 项目根目录，Pathlib使得路径操作更简单、跨平台
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 数据相关路径
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"          # 存放所有原始 .tif 图像 
LABELED_DATA_DIR = DATA_DIR / "labeled"    # 存放已标注的图像和掩码 
UNLABELED_DATA_DIR = DATA_DIR / "unlabeled"  # 存放未标注图像的目录 
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
NUM_CLASSES = 4

# U-Net解码器上采样时是否使用双线性插值 
BILINEAR = True


MAPPING = {
	(126, 126, 255): 0, #AM
	(255, 255, 0): 1, #SE
	(0, 255, 0): 2, #carbon
	(255, 0, 0): 3 #void
}
COLORS = [(126, 126, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

# --- 4. JSON文件命名 (JSON Filename Configuration) ---
TRAIN_JSON_FILENAME = "train_dataset.json"
VALID_JSON_FILENAME = "valid_dataset.json"
TEST_JSON_FILENAME = "test_dataset.json"
MASTER_UNLABELED_JSON_FILENAME = "master_unlabeled_dataset.json"
MASTER_LABELED_JSON_FILENAME = "master_labeled_dataset.json"