import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Union

# --- 动态添加项目根目录到Python路径 ---
try:
    project_root = Path(__file__).resolve().parent.parent
except NameError:
    project_root = Path.cwd()

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# --- 导入项目模块和配置 ---
try:
    from configs import base as cfg_base
    from configs.inference import inference_config as cfg_inference
    from models.segmentation_unet import SegmentationUNet
    from utils.augmentation_util import load_dataset_stats
except ImportError as e:
    print(f"错误: 导入模块失败。请确保此脚本位于 'tasks' 文件夹下，且项目结构正确。")
    print(f"具体错误: {e}")
    sys.exit(1)


def create_model(encoder_name: str, decoder_name: str) -> nn.Module:
    """
    根据配置动态创建编码器和解码器，并组装成一个分割模型。
    与训练任务中的版本保持一致。
    """
    print(f"--- 正在创建模型: Encoder: {encoder_name}, Decoder: {decoder_name} ---")

    if encoder_name == 'unet':
        from models.encoders.unet_encoder import UNetEncoder
        encoder = UNetEncoder(n_channels=cfg_base.INPUT_DEPTH)
    # 在这里可以继续添加其他编码器的 'elif' 分支
    else:
        raise ValueError(f"未知的编码器名称: '{encoder_name}'")

    encoder_channels = encoder.get_channels()
    if decoder_name == 'unet':
        from models.decoders.unet_decoder import UNetDecoder
        decoder = UNetDecoder(encoder_channels, n_classes=cfg_base.NUM_CLASSES)
    # 在这里可以继续添加其他解码器的 'elif' 分支
    else:
        raise ValueError(f"未知的解码器名称: '{decoder_name}'")

    model = SegmentationUNet(encoder, decoder)
    return model


def build_inference_transforms(height: int, width: int):
    """构建用于推理时的数据预处理流水线。"""
    mean, std = load_dataset_stats(expected_input_depth=cfg_base.INPUT_DEPTH)
    
    if not mean or not std:
        print("警告: 未找到数据集统计数据，将不进行归一化。")
        return A.Compose([
            A.Resize(height=height, width=width), # PIL.Image.Resampling is not needed here
            ToTensorV2(),
        ])
        
    return A.Compose([
        A.Resize(height=height, width=width),
        A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
        ToTensorV2(),
    ])

def class_to_rgb(pred_mask: np.ndarray) -> np.ndarray:
    """将单通道的类别掩码转换为彩色的RGB图像。"""
    h, w = pred_mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 使用base中的COLOR_MAPPING
    for class_idx, color in cfg_base.COLOR_MAPPING.items():
        rgb_mask[pred_mask == class_idx] = color
    return rgb_mask

def main_inference():
    """主推理函数。"""
    print("--- 开始推理任务 ---")
    
    device = torch.device(cfg_inference.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    model_path = project_root / cfg_inference.MODEL_CHECKPOINT_PATH
    input_dir = project_root / cfg_inference.INPUT_DIR
    output_dir = project_root / cfg_inference.OUTPUT_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")

    # **关键修正**: 使用 create_model 函数动态创建模型
    model = create_model(
        encoder_name=cfg_inference.ENCODER_NAME,
        decoder_name=cfg_inference.DECODER_NAME
    ).to(device)
    
    if not model_path.exists():
        print(f"错误: 模型检查点未找到 -> {model_path}")
        return
        
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("模型加载成功。")

    transform = build_inference_transforms(cfg_base.IMAGE_HEIGHT, cfg_base.IMAGE_WIDTH)

    image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']
    image_files = [p for p in input_dir.glob('*') if p.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"错误: 在输入目录 {input_dir} 中未找到任何图像文件。")
        return

    print(f"找到 {len(image_files)} 张图像进行处理。")

    with torch.no_grad():
        for image_path in tqdm(image_files, desc="Processing images"):
            try:
                image_pil = Image.open(image_path).convert('L')
                image_np = np.array(image_pil, dtype=np.float32) / 255.0
                image_stack = np.stack([image_np] * cfg_base.INPUT_DEPTH, axis=-1)
                augmented = transform(image=image_stack)
                image_tensor = augmented['image'].unsqueeze(0).to(device)

                logits = model(image_tensor)
                
                original_size = image_pil.size[::-1]
                logits_resized = F.interpolate(logits, size=original_size, mode='bilinear', align_corners=False)
                pred_mask = torch.argmax(logits_resized, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

                rgb_mask = class_to_rgb(pred_mask)
                output_path = output_dir / f"{image_path.stem}_mask.png"
                Image.fromarray(rgb_mask).save(output_path)
            except Exception as e:
                print(f"\n处理文件 {image_path.name} 时出错: {e}")

    print("\n--- 推理任务完成 ---")
    print(f"所有预测结果已保存至: {output_dir}")

if __name__ == '__main__':
    main_inference()