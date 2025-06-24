from pathlib import Path

# --- JSON Generator Specific Configurations ---

#"data/raw/"
RAW_IMAGE_SOURCE_DIR_NAME = "data/raw"

#这是 "data/labeled/"
LABELED_MASK_DIR_NAME = "data/labeled"

#这是 "json/"
JSON_OUTPUT_DIR_NAME = "json"

# --- 数据集参数 (可以被命令行覆盖) ---
INPUT_DEPTH = 3  # 模型期望的输入深度 (堆叠帧数)

FILENAME_PATTERN_STR = r"^(.*?)[\-_]?(\d+)\.(?:png|tif|jpg|jpeg)$"

# 图像和掩码文件的预期扩展名
EXPECTED_IMAGE_EXTENSIONS = ['.png', '.tif', '.jpg', '.jpeg']
EXPECTED_MASK_EXTENSIONS = ['.png']  # 掩码通常是 .png 格式

# --- 数据集分割参数 (用于 split_labeled 和 generate_all 模式) ---
DEFAULT_VAL_SPLIT = 0.15
DEFAULT_TEST_SPLIT = 0.15
DEFAULT_RANDOM_SEED = 42