import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from easydict import EasyDict as edict
from PIL import Image
from pathlib import Path
import json
from sklearn.metrics import jaccard_score

# 导入配置
import configs.base as base_config_module
from configs.inference.evaluation_config import get_evaluation_config
from datasets.sem_datasets import SemSegmentationDataset
from models.segmentation_unet import SegmentationUNet

# 辅助函数
def build_base_config_from_module(module):
    """根据 base.py 的结构构建 EasyDict 配置对象。"""
    print("正在从 'configs/base.py' 加载配置变量...")
    try:
        base_config = edict()
        base_config.IMAGE_SIZE = (module.IMAGE_HEIGHT, module.IMAGE_WIDTH)
        base_config.INPUT_DEPTH = module.INPUT_DEPTH
        base_config.NUM_CLASSES = module.NUM_CLASSES
        base_config.MAPPING = getattr(module, 'MAPPING', {})
        base_config.CLASS_NAMES = getattr(module, 'CLASS_NAMES', ['class_%d' % i for i in range(base_config.NUM_CLASSES)])
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

def strip_mask_suffix(name):
    for suf in ['_mask', '-mask', '.mask']:
        if name.endswith(suf):
            return name[:-len(suf)]
    return name

def load_mask(mask_path, base_config):
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mapping = base_config.MAPPING
        out = np.zeros(mask.shape[:2], dtype=np.int64)
        for rgb, idx in mapping.items():
            out[(mask == rgb).all(axis=-1)] = idx
        return out
    return mask

def dice_score(pred, gt, num_classes):
    scores = []
    for i in range(num_classes):
        pred_i = (pred == i)
        gt_i = (gt == i)
        intersection = (pred_i & gt_i).sum()
        union = pred_i.sum() + gt_i.sum()
        if union == 0:
            scores.append(1.0)
        else:
            scores.append(2.0 * intersection / (union + 1e-6))
    return scores

def iou_score(pred, gt, num_classes):
    scores = []
    for i in range(num_classes):
        pred_i = (pred == i)
        gt_i = (gt == i)
        intersection = (pred_i & gt_i).sum()
        union = (pred_i | gt_i).sum()
        if union == 0:
            scores.append(1.0)
        else:
            scores.append(intersection / (union + 1e-6))
    return scores

# 核心函数
def evaluate_model(config, base_config):
    """统一的评估函数：进行预测、可视化和指标计算。"""
    print("\n--- 开始模型评估任务 ---")
    
    device = get_device()
    output_dir = config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    print(f"结果将保存至: {output_dir}")
    
    print("加载验证数据集...")
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
        raise FileNotFoundError(f"找不到模型权重文件: {checkpoint_path}")
    
    print(f"从 {checkpoint_path} 加载权重...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    
    num_to_inspect = config.NUM_IMAGES_TO_INSPECT if config.NUM_IMAGES_TO_INSPECT > 0 else len(val_loader)
    
    dice_all = []
    iou_all = []
    lines = []
    
    with torch.no_grad():
        for i, (images, true_masks) in enumerate(tqdm(val_loader, total=num_to_inspect, desc="评估进度")):
            if i >= num_to_inspect: break
            
            images = images.to(device, dtype=torch.float32)
            pred_logits = model(images)
            pred_probs = torch.sigmoid(pred_logits)
            pred_masks = (pred_probs > 0.5).float()
            
            image_np = images.cpu().numpy().squeeze()
            true_mask_np = true_masks.cpu().numpy().squeeze()
            pred_mask_np = pred_masks.cpu().numpy().squeeze()
            
            if image_np.ndim == 3:
                s = image_np.shape[0] // 2
                image_np = image_np[s]
                true_mask_np = true_mask_np[s]
                pred_mask_np = pred_mask_np[s]
            
            # 可视化
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(f'Sample {i+1}', fontsize=16)
            axes[0].imshow(image_np, cmap='gray'); axes[0].set_title('Original Image'); axes[0].axis('off')
            axes[1].imshow(true_mask_np, cmap='gray'); axes[1].set_title('Ground Truth Mask'); axes[1].axis('off')
            axes[2].imshow(pred_mask_np, cmap='gray'); axes[2].set_title('Model Prediction'); axes[2].axis('off')
            plt.savefig(os.path.join(output_dir, f'evaluation_sample_{i+1:04d}.png'), bbox_inches='tight', dpi=150)
            plt.close(fig)
            
            # 指标计算
            if pred_mask_np.shape != true_mask_np.shape:
                pred_mask_np = np.array(Image.fromarray(pred_mask_np.astype(np.uint8)).resize(true_mask_np.shape[::-1], resample=Image.NEAREST))
            
            dice = dice_score(pred_mask_np, true_mask_np, base_config.NUM_CLASSES)
            iou = iou_score(pred_mask_np, true_mask_np, base_config.NUM_CLASSES)
            dice_all.append(dice)
            iou_all.append(iou)
            
            line = f"Sample {i+1}: "
            for j, cname in enumerate(base_config.CLASS_NAMES):
                line += f" {cname} Dice: {dice[j]:.4f} IoU: {iou[j]:.4f}"
            print(line)
            lines.append(line)
    
    # 汇总指标
    dice_all = np.array(dice_all)
    iou_all = np.array(iou_all)
    lines.append("\n=== 总体均值 ===")
    for j, cname in enumerate(base_config.CLASS_NAMES):
        s = f"{cname} Dice: {dice_all[:,j].mean():.4f} IoU: {iou_all[:,j].mean():.4f}"
        print(s)
        lines.append(s)
    print(f"平均Dice: {dice_all.mean():.4f}")
    print(f"平均IoU: {iou_all.mean():.4f}")
    lines.append(f"平均Dice: {dice_all.mean():.4f}")
    lines.append(f"平均IoU: {iou_all.mean():.4f}")
    
    # 保存结果
    result_save_path = config.RESULT_SAVE_PATH or os.path.join(output_dir, 'evaluation_results.txt')
    with open(result_save_path, 'w', encoding='utf-8') as f:
        for l in lines:
            f.write(l + '\n')
    print(f"评估结果已保存到: {result_save_path}")
    
    print(f"\n评估完成！生成 {num_to_inspect} 张对比图和指标报告。")

# 主程序入口
if __name__ == '__main__':
    print("--- 正在初始化评估脚本 ---")
    base_config = build_base_config_from_module(base_config_module)
    evaluation_config = get_evaluation_config()
    evaluate_model(evaluation_config, base_config) 