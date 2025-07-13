# self_segmentation/configs/inference/inference_config.py
from pathlib import Path

# --- 推理任务配置 ---

# ==============================================================================
# --- 编码器 (Encoder) 配置 ---
# !! 这里的设置必须与您要加载的检查点所对应的模型架构完全一致 !!
# 可用选项: 'unet', 'resnet50', 'efficientnet-b4', 'dinov2'
ENCODER_NAME = "unet"

# --- 解码器 (Decoder) 配置 ---
# !! 这里的设置必须与您要加载的检查点所对应的模型架构完全一致 !!
# 可用选项: 'unet', 'deeplab'
DECODER_NAME = "unet"
# ==============================================================================


# 1. 模型检查点路径
#    指定要用于推理的、已经训练好的模型权重文件 (.pth)。
#    请确保这个权重文件是上面指定的模型架构训练得出的。
#    例如: "models/checkpoints/sem_segmentation_from_scratch_unet_unet/best_model.pth"
MODEL_CHECKPOINT_PATH = "models/checkpoints/sem_segmentation_from_scratch_unet_unet/best_model.pth"

# 2. 输入图像目录
#    存放您想要进行分割的、新的原始SEM图像的文件夹。
INPUT_DIR = "data/inference/input"

# 3. 输出掩码目录
#    用于保存模型生成的彩色分割掩码图像的文件夹。
OUTPUT_DIR = "data/inference/output"

# 4. 推理设备
#    'cuda' 或 'cpu'。
DEVICE = "cuda"