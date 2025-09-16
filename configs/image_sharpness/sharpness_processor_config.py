"""
图像清晰度处理器配置文件
主要功能：
1. 计算每张图片的清晰度，生成分析图表和JSON文件
2. 对图像清晰度进行平均化处理，让所有图像的清晰度基本相等
3. 将处理后的图像保存到各自数据集目录下
4. 为新的数据集生成JSON文件用于训练
"""

from pathlib import Path

# ===========================================
# 输入输出路径配置
# ===========================================

# 输入输出路径 - 修改为指向datasets目录下的原始图像文件夹
INPUT_FOLDER = Path("datasets/dataset1_LInCl/raw_images")  # 修改：指向datasets目录下的原始图像
OUTPUT_FOLDER = Path("data/sharpened")
JSON_OUTPUT_PATH = Path("json/sharpness_analysis.json")  # 修改：将JSON输出到data目录

# 数据集JSON路径配置 - 修改为指向datasets目录下的原始图像文件夹
DATASET_JSON_PATH = Path("json/master_labeled_dataset.json")
USE_DATASET_JSON = False  # 修改：改为False，优先使用文件夹扫描模式
DATASET_IMAGE_FOLDER = Path("datasets/dataset1_LInCl/raw_images")  # 修改：指向datasets目录下的原始图像文件夹

# 多数据集输入路径配置
# DATASET_INPUT_PATHS = {
#     "dataset1_LInCl": Path("datasets/dataset1_LInCl/raw_images"),
#     "dataset2_LPSCl": Path("datasets/dataset2_LPSCl/raw_images"),
#     "dataset3_LNOCl": Path("datasets/dataset3_LNOCl/raw_images")
# }

# 数据集名称列表 - 便于后续增加新数据集
DATASET_NAMES = [
    "dataset1_LInCl",
    "dataset2_LPSCl", 
    "dataset3_LNOCl"
]

# 辅助函数：根据数据集名称列表动态生成配置
def generate_dataset_configs():
    """根据数据集名称列表动态生成输入输出路径配置"""
    input_paths = {}
    output_paths = {}
    
    for dataset_name in DATASET_NAMES:
        # 生成输入路径（原始图像）
        input_paths[dataset_name] = Path(f"datasets/{dataset_name}/raw_images")
        # 生成输出路径（清晰度平均化后的图像）
        output_paths[dataset_name] = Path(f"datasets/{dataset_name}/sharpness_averaged")
    
    return input_paths, output_paths

# 动态生成数据集配置
DATASET_INPUT_PATHS, DATASET_OUTPUT_PATHS = generate_dataset_configs()

# ===========================================
# 清晰度评估配置
# ===========================================

# 清晰度评估方法
SHARPNESS_METHODS = ["lapvar", "tenengrad", "fft_energy"]
TARGET_SHARPNESS = 0.5
SHARPNESS_TOLERANCE = 0.1

# ===========================================
# 清晰度平均化配置
# ===========================================

# 清晰度平均化相关配置
ENABLE_SHARPNESS_AVERAGING = True
AVERAGING_METHOD = "histogram_matching"
AVERAGING_TARGET = "mean"
CUSTOM_TARGET_SHARPNESS = 0.6

# 输出目录配置
SHARPNESS_AVERAGED_OUTPUT_DIR = "sharpness_averaged"
GENERATE_TRAINING_JSON = True
GENERATE_SSL_JSON = True

# 数据集输出路径配置（保留原有配置作为备用）
# DATASET_OUTPUT_PATHS = {
#     "dataset1_LInCl": Path("datasets/dataset1_LInCl/sharpness_averaged"),
#     "dataset2_LPSCl": Path("datasets/dataset2_LPSCl/sharpness_averaged"),
#     "dataset3_LNOCl": Path("datasets/dataset3_LNOCl/sharpness_averaged")
# }

# ===========================================
# 图像处理配置
# ===========================================

# 支持的图像格式
SUPPORTED_FORMATS = [".png", ".jpg", ".jpeg", ".tif", ".bmp"]
BATCH_SIZE = 4

# ===========================================
# 图表输出配置
# ===========================================

CHART_OUTPUT_FOLDER = Path("data/charts")  # 修改：将图表输出到data目录
GENERATE_BEFORE_CHARTS = True
GENERATE_AFTER_CHARTS = True

# 创建必要的目录
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
CHART_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
JSON_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# 创建数据集输出目录
for dataset_path in DATASET_OUTPUT_PATHS.values():
    dataset_path.mkdir(parents=True, exist_ok=True)

# 检查数据集JSON文件是否存在
if USE_DATASET_JSON and not DATASET_JSON_PATH.exists():
    print(f"警告: 数据集JSON文件不存在: {DATASET_JSON_PATH}")
    USE_DATASET_JSON = False
