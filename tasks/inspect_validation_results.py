import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from easydict import EasyDict as edict

# --- 模块导入区 ---
import configs.base as base_config_module
from configs.inference.inspection_config import get_inspection_config
from datasets.sem_datasets import SemSegmentationDataset
from models.segmentation_unet import SegmentationUNet

# --- 辅助函数区 ---

def build_base_config_from_module(module):
    """根据您 base.py 的实际结构，手动构建一个 EasyDict 配置对象。"""
    print("正在从 'configs/base.py' 加载配置变量...")
    try:
        base_config = edict()
        base_config.IMAGE_SIZE = (module.IMAGE_HEIGHT, module.IMAGE_WIDTH)
        base_config.INPUT_DEPTH = module.INPUT_DEPTH
        base_config.NUM_CLASSES = module.NUM_CLASSES
        print("成功构建基础配置对象。")
        return base_config
    except AttributeError as e:
        print(f"错误: 在 'configs/base.py' 中找不到必要的配置变量。缺失了: {e}")
        exit()

def get_device():
    """获取计算设备 (GPU/CPU)"""
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"设备检测: {'CUDA (GPU)' if use_cuda else 'CPU'} 将被使用。")
    return device

# --- 核心功能函数 ---

def inspect_validation_set(config, base_config):
    """加载模型，对验证集进行预测，并保存可视化结果。"""
    print("\n--- 开始验证集检查任务 (最终版) ---")
    
    device = get_device()
    output_dir = config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    print(f"检查结果将保存至: {output_dir}")

    print("加载验证数据集...")
    # [修正] 使用正确的参数名 file_list 替换 json_path
    val_dataset = SemSegmentationDataset(
        file_list=config.DATA_LOADER_CONFIG.val.file_list,
        image_size=tuple(base_config.IMAGE_SIZE),
        is_train=False
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=config.DATA_LOADER_CONFIG.val.num_workers)

    print("加载模型...")
    model = SegmentationUNet(
        encoder_name=config.MODEL_CONFIG.encoder_name,
        decoder_name=config.MODEL_CONFIG.decoder_name,
        in_channels=base_config.INPUT_DEPTH,
        num_classes=base_config.NUM_CLASSES
    )
    
    checkpoint_path = config.CHECKPOINT_PATH
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"关键错误: 找不到模型权重文件 -> {checkpoint_path}")

    print(f"从 {checkpoint_path} 加载权重...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    print("开始遍历验证集并生成对比图...")
    num_to_inspect = config.NUM_IMAGES_TO_INSPECT if config.NUM_IMAGES_TO_INSPECT > 0 else len(val_loader)
    
    with torch.no_grad():
        for i, (images, true_masks) in enumerate(tqdm(val_loader, total=num_to_inspect, desc="生成检查图片")):
            if i >= num_to_inspect: break
            
            images = images.to(device, dtype=torch.float32)
            pred_logits = model(images)
            pred_probs = torch.sigmoid(pred_logits)
            pred_masks = (pred_probs > 0.5).float()

            image_np, true_mask_np, pred_mask_np = images.cpu().numpy().squeeze(), true_masks.cpu().numpy().squeeze(), pred_masks.cpu().numpy().squeeze()
            
            if image_np.ndim == 3:
                s = image_np.shape[0] // 2
                image_np, true_mask_np, pred_mask_np = image_np[s], true_mask_np[s], pred_mask_np[s]

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(f'Sample {i+1}', fontsize=16)
            axes[0].imshow(image_np, cmap='gray'); axes[0].set_title('Original Image'); axes[0].axis('off')
            axes[1].imshow(true_mask_np, cmap='gray'); axes[1].set_title('Ground Truth Mask'); axes[1].axis('off')
            axes[2].imshow(pred_mask_np, cmap='gray'); axes[2].set_title('Model Prediction'); axes[2].axis('off')
            
            plt.savefig(os.path.join(output_dir, f'inspection_sample_{i+1:04d}.png'), bbox_inches='tight', dpi=150)
            plt.close(fig)

    print(f"\n检查完成！成功生成 {num_to_inspect} 张对比图，保存在 {output_dir} 文件夹中。")

# --- 主程序入口 ---
if __name__ == '__main__':
    print("--- 正在初始化检查脚本 ---")
    
    base_config = build_base_config_from_module(base_config_module)
    inspection_config = get_inspection_config()
    inspect_validation_set(inspection_config, base_config)