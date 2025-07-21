from pathlib import Path

# --- JSON Generator Specific Configurations ---

# 原始图像目录（用于json生成，改为raw_images）
RAW_IMAGE_SOURCE_DIR_NAME = "raw_images"

# 掩码目录（用于json生成，改为masks_3class）
LABELED_MASK_DIR_NAME = "masks_3class"

#这是 "json/"
JSON_OUTPUT_DIR_NAME = "json"

# --- 数据集参数 (可以被命令行覆盖) ---
INPUT_DEPTH = 3  # 模型期望的输入深度 (堆叠帧数)

FILENAME_PATTERN_STR = r"^(.*?)[\-_]?(\d+)\.(?:png|tif|jpg|jpeg)$"

# 图像和掩码文件的预期扩展名
EXPECTED_IMAGE_EXTENSIONS = ['.png', '.tif', '.jpg', '.jpeg']
EXPECTED_MASK_EXTENSIONS = ['.png']  # 掩码通常是 .png 格式