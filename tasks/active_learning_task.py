# tasks/active_learning_task.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path
import sys
import os
import json
from tqdm import tqdm
import shutil
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 添加项目根目录到sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from datasets.sem_datasets import SemSegmentationDataset
from models.segmentation_unet import SegmentationUNet
from utils.training_util import get_loss_function, train_one_epoch, evaluate_model
from utils.logging_util import Logger
from utils.augmentation_util import load_dataset_stats
from utils.uncertainty_util import compute_combined_uncertainty, detect_prediction_boundaries
from configs.active_learning import active_learning_config as cfg_al
from configs import base as cfg_base

class UnlabeledDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        # 修复路径处理，确保跨平台兼容
        if isinstance(image_dir, str):
            # 如果是字符串路径，转换为Path对象
            image_dir = Path(image_dir)
        self.image_dir = project_root / image_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith(('.png', '.jpg'))])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_dir / self.image_files[idx]
        image = np.array(Image.open(img_path).convert('L'), dtype=np.float32) / 255.0
        image_stack = np.stack([image] * cfg_base.INPUT_DEPTH, axis=-1)
        if self.transform:
            augmented = self.transform(image=image_stack)
            image = augmented['image']
        return image, self.image_files[idx]

def select_samples(uncertainties, image_files, k):
    sorted_indices = torch.argsort(uncertainties, descending=True)
    selected = [(image_files[i], uncertainties[i].item()) for i in sorted_indices[:k]]
    return selected

def save_checkpoint(model, optimizer, epoch, train_loss, val_dice, checkpoint_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_dice': val_dice,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"检查点已保存到: {checkpoint_path}")

def load_checkpoint(model, optimizer, checkpoint_path, device):
    """加载训练检查点"""
    if not os.path.exists(checkpoint_path):
        print(f"检查点文件不存在: {checkpoint_path}")
        return 0, float('inf'), 0.0
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']
    val_dice = checkpoint['val_dice']
    
    print(f"从检查点恢复训练 - Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Dice: {val_dice:.4f}")
    return epoch, train_loss, val_dice

def class_to_rgb(pred_mask: np.ndarray) -> np.ndarray:
    """将单通道的类别掩码转换为彩色的RGB图像。"""
    h, w = pred_mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 使用base中的COLOR_MAPPING
    for class_idx, color in cfg_base.COLOR_MAPPING.items():
        rgb_mask[pred_mask == class_idx] = color
    return rgb_mask

def main_active_learning():
    print("--- 开始主动学习任务 ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 显示初始状态
    initial_labeled_count = len([f for f in os.listdir(project_root / cfg_al.LABELED_DIR) if f.endswith('.png')])
    initial_unlabeled_count = len([f for f in os.listdir(project_root / cfg_al.UNLABELED_POOL_DIR) if f.endswith('.png')])
    print(f"初始已标注图像数量: {initial_labeled_count}")
    print(f"初始无标签图像数量: {initial_unlabeled_count}")
    print(f"主动学习配置: {cfg_al.NUM_ITERATIONS} 轮迭代，每轮标注 {cfg_al.SAMPLES_PER_ITER} 张图像")

    # 创建目录
    (project_root / cfg_al.PREDICTIONS_DIR).mkdir(parents=True, exist_ok=True)
    (project_root / cfg_al.UNCERTAINTY_MAPS_DIR).mkdir(parents=True, exist_ok=True)
    (project_root / cfg_al.SELECTION_INFO_DIR).mkdir(parents=True, exist_ok=True)
    cfg_al.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # 使用与inference相同的模型创建方式
    def create_model(encoder_name: str, decoder_name: str) -> nn.Module:
        """
        根据配置动态创建编码器和解码器，并组装成一个分割模型。
        与inference任务保持一致。
        """
        print(f"--- 正在创建模型: Encoder: {encoder_name}, Decoder: {decoder_name} ---")

        if encoder_name == 'unet':
            from models.encoders.unet_encoder import UNetEncoder
            encoder = UNetEncoder(n_channels=cfg_base.INPUT_DEPTH)
        else:
            raise ValueError(f"未知的编码器名称: '{encoder_name}'")

        encoder_channels = encoder.get_channels()
        if decoder_name == 'unet':
            from models.decoders.unet_decoder import UNetDecoder
            decoder = UNetDecoder(encoder_channels, n_classes=cfg_base.NUM_CLASSES)
        else:
            raise ValueError(f"未知的解码器名称: '{decoder_name}'")

        model = SegmentationUNet(encoder, decoder)
        return model

    # 创建模型
    model = create_model(
        encoder_name=cfg_al.ENCODER_NAME,
        decoder_name=cfg_al.DECODER_NAME
    ).to(device)

    # 加载预训练模型
    if cfg_al.USE_PRETRAINED_MODEL and cfg_al.PRETRAINED_MODEL_PATH.exists():
        print(f"加载预训练模型: {cfg_al.PRETRAINED_MODEL_PATH}")
        print(f"模型文件大小: {cfg_al.PRETRAINED_MODEL_PATH.stat().st_size / (1024*1024):.2f} MB")
        
        checkpoint = torch.load(cfg_al.PRETRAINED_MODEL_PATH, map_location=device)
        
        # 详细输出checkpoint信息
        print(f"\n=== Checkpoint详细信息 ===")
        print(f"Checkpoint类型: {type(checkpoint)}")
        if isinstance(checkpoint, dict):
            print(f"Checkpoint键: {list(checkpoint.keys())}")
            for key, value in checkpoint.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: tensor shape={value.shape}, dtype={value.dtype}")
                elif isinstance(value, (int, float)):
                    print(f"  {key}: {value}")
                elif isinstance(value, dict):
                    print(f"  {key}: dict with {len(value)} items")
                else:
                    print(f"  {key}: {type(value)}")
        
        # 检查checkpoint格式并加载
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("\n✓ 成功加载模型权重 (model_state_dict)")
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            print("\n✓ 成功加载模型权重 (state_dict)")
        else:
            # 假设checkpoint直接是state_dict
            model.load_state_dict(checkpoint)
            print("\n✓ 成功加载模型权重 (direct state_dict)")
        
        # 显示预训练模型详细信息
        print(f"\n=== 预训练模型信息 ===")
        if 'epoch' in checkpoint:
            print(f"训练轮数: {checkpoint['epoch']}")
        if 'val_dice' in checkpoint:
            print(f"验证Dice: {checkpoint['val_dice']:.4f}")
        if 'train_loss' in checkpoint:
            print(f"训练损失: {checkpoint['train_loss']:.4f}")
        if 'optimizer_state_dict' in checkpoint:
            print(f"包含优化器状态: 是")
        
        # 验证模型加载
        print(f"\n=== 模型验证 ===")
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
    else:
        print("使用随机初始化的模型进行主动学习")
        if not cfg_al.PRETRAINED_MODEL_PATH.exists():
            print(f"预训练模型文件不存在: {cfg_al.PRETRAINED_MODEL_PATH}")
        if not cfg_al.USE_PRETRAINED_MODEL:
            print("配置中禁用了预训练模型加载")

    # 使用与inference相同的数据预处理
    def build_inference_transforms(height: int, width: int):
        """构建用于推理时的数据预处理流水线。"""
        mean, std = load_dataset_stats(expected_input_depth=cfg_base.INPUT_DEPTH)
        
        if not mean or not std:
            print("警告: 未找到数据集统计数据，将不进行归一化。")
            return A.Compose([
                A.Resize(height=height, width=width),
                ToTensorV2(),
            ])
            
        return A.Compose([
            A.Resize(height=height, width=width),
            A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
            ToTensorV2(),
        ])

    transform = build_inference_transforms(cfg_base.IMAGE_HEIGHT, cfg_base.IMAGE_WIDTH)

    # 初始化日志记录器
    logger = Logger(cfg_al.log_config)
    
    # 跟踪性能改进
    best_overall_dice = 0.0
    performance_history = []

    for iter_num in range(1, cfg_al.NUM_ITERATIONS + 1):
        print(f"\n=== 主动学习迭代 {iter_num} ===")

        # 步骤1: 对无标签池进行推理
        unlabeled_dataset = UnlabeledDataset(cfg_al.UNLABELED_POOL_DIR, transform=transform)
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=cfg_al.BATCH_SIZE, shuffle=False)

        model.eval()
        uncertainties = []
        image_files = []
        predictions = {}  # 存储预测结果
        
        with torch.no_grad():
            for images, files in tqdm(unlabeled_loader, desc="Inferring unlabeled pool"):
                images = images.to(device)
                logits = model(images)
                
                # 计算预测掩码用于边界不确定性
                pred_masks = torch.argmax(logits, dim=1)
                
                # 使用边界不确定性计算
                unc = compute_combined_uncertainty(
                    logits, 
                    cfg_al.UNCERTAINTY_METHODS,
                    prediction_mask=pred_masks,
                    dilation_size=cfg_al.BOUNDARY_DILATION_SIZE
                )
                uncertainties.extend(unc.cpu())
                image_files.extend(files)
                
                # 存储预测结果
                pred_masks_np = pred_masks.cpu().numpy()
                
                for i, file_name in enumerate(files):
                    predictions[file_name] = pred_masks_np[i]

        uncertainties = torch.tensor(uncertainties)

        # 步骤2: 选择样本
        selected = select_samples(uncertainties, image_files, cfg_al.SAMPLES_PER_ITER)
        selection_info_path = project_root / cfg_al.SELECTION_INFO_DIR / f"iteration_{iter_num}_selected.json"
        with open(selection_info_path, 'w') as f:
            json.dump(selected, f, indent=4)
        
        # 显示选择统计信息
        print(f"\n=== 样本选择统计 ===")
        print(f"无标签池总图像数: {len(image_files)}")
        print(f"本轮选择图像数: {len(selected)}")
        print(f"使用的不确定性方法: {cfg_al.UNCERTAINTY_METHODS}")
        print(f"边界膨胀像素数: {cfg_al.BOUNDARY_DILATION_SIZE}")
        
        if len(selected) > 0:
            selected_uncertainty = selected[0][1]
            print(f"选中图像不确定性分数: {selected_uncertainty:.4f}")
            print(f"不确定性分数范围: {uncertainties.min():.4f} - {uncertainties.max():.4f}")
            
            # 分析选中图像的边界信息
            selected_img = selected[0][0]
            if selected_img in predictions:
                pred_mask = predictions[selected_img]
                boundaries = detect_prediction_boundaries(pred_mask, cfg_al.BOUNDARY_DILATION_SIZE)
                boundary_area = np.sum(boundaries)
                total_area = pred_mask.size
                boundary_ratio = boundary_area / total_area * 100
                print(f"选中图像的边界分析:")
                print(f"  - 边界区域像素数: {boundary_area}")
                print(f"  - 边界区域占比: {boundary_ratio:.2f}%")
                print(f"  - 预测类别分布: {np.unique(pred_mask)}")
        
        print(f"选择信息已保存到: {selection_info_path}")

        # 步骤3: 生成预测mask图像
        selected_img, uncertainty_score = selected[0]  # 只取第一张图像
        print(f"\n=== 为选中图像生成预测mask ===")
        print(f"图像名称: {selected_img}")
        print(f"不确定性分数: {uncertainty_score:.4f}")
        
        # 获取预测结果
        if selected_img in predictions:
            pred_mask = predictions[selected_img]
            
            # 重新加载选中图像进行详细分析
            selected_dataset = UnlabeledDataset(cfg_al.UNLABELED_POOL_DIR, transform=transform)
            selected_idx = None
            for idx, (_, files) in enumerate(selected_dataset):
                if files == selected_img:
                    selected_idx = idx
                    break
            
            if selected_idx is not None:
                # 使用与inference完全相同的预测逻辑
                selected_loader = DataLoader(selected_dataset, batch_size=1, shuffle=False)
                selected_data = list(selected_loader)[selected_idx]
                selected_tensor = selected_data[0].to(device)
                
                with torch.no_grad():
                    # 使用inference的预测逻辑
                    logits = model(selected_tensor)
                    
                    # 获取原始图像尺寸
                    original_image_path = project_root / cfg_al.UNLABELED_POOL_DIR / selected_img
                    original_image_pil = Image.open(original_image_path).convert('L')
                    original_size = original_image_pil.size[::-1]  # (height, width)
                    
                    # 使用inference的尺寸调整和预测逻辑
                    logits_resized = F.interpolate(logits, size=original_size, mode='bilinear', align_corners=False)
                    
                    # 添加详细的logits分析
                    print(f"\n=== Logits分析 ===")
                    print(f"Logits形状: {logits_resized.shape}")
                    print(f"Logits值范围: {logits_resized.min():.4f} - {logits_resized.max():.4f}")
                    
                    # 计算概率分布
                    probs = F.softmax(logits_resized, dim=1)
                    print(f"概率分布形状: {probs.shape}")
                    print(f"概率值范围: {probs.min():.4f} - {probs.max():.4f}")
                    
                    # 分析每个类别的概率
                    for class_id in range(probs.shape[1]):
                        class_probs = probs[0, class_id]
                        mean_prob = class_probs.mean().item()
                        max_prob = class_probs.max().item()
                        min_prob = class_probs.min().item()
                        class_name = {0: "carbon", 1: "SE", 2: "AM"}[class_id]
                        print(f"  {class_name}类别: 平均={mean_prob:.4f}, 最大={max_prob:.4f}, 最小={min_prob:.4f}")
                    
                    pred_mask = torch.argmax(logits_resized, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                    
                    # 分析预测结果
                    unique_preds = np.unique(pred_mask)
                    print(f"\n=== 预测结果分析 ===")
                    print(f"预测掩码中的唯一值: {unique_preds}")
                    print(f"预测掩码形状: {pred_mask.shape}")
                    print(f"预测掩码数据类型: {pred_mask.dtype}")
                    print(f"预测掩码值范围: {pred_mask.min()} - {pred_mask.max()}")
                    
                    for pred_val in unique_preds:
                        count = np.sum(pred_mask == pred_val)
                        percentage = (count / pred_mask.size) * 100
                        class_name = {0: "carbon", 1: "SE", 2: "AM"}.get(pred_val, f"未知{pred_val}")
                        print(f"  - {class_name} (ID: {pred_val}): {count} 像素 ({percentage:.2f}%)")
                    
                    # 检查是否所有类别都被预测
                    expected_classes = [0, 1, 2]
                    missing_classes = [cls for cls in expected_classes if cls not in unique_preds]
                    if missing_classes:
                        print(f"⚠️  警告: 以下类别未被预测: {missing_classes}")
                        for missing_cls in missing_classes:
                            class_name = {0: "carbon", 1: "SE", 2: "AM"}[missing_cls]
                            print(f"    - {class_name} (ID: {missing_cls})")
                    else:
                        print("✓ 所有类别都被预测到")

                    # 生成RGB mask
                    rgb_mask = class_to_rgb(pred_mask)
                    
                    # 创建迭代目录
                    iteration_dir = project_root / cfg_al.PREDICTIONS_DIR / f"iteration_{iter_num}"
                    iteration_dir.mkdir(parents=True, exist_ok=True)
                    
                    # 保存预测mask图像到迭代目录
                    mask_filename = f"{selected_img.split('.')[0]}_prediction_mask.png"
                    mask_path = iteration_dir / mask_filename
                    
                    mask_pil = Image.fromarray(rgb_mask)
                    mask_pil.save(mask_path)
                    print(f"预测mask图像已保存: {mask_path}")
                    
                    # 复制原始图像到迭代目录
                    original_image_path = project_root / cfg_al.UNLABELED_POOL_DIR / selected_img
                    original_copy_path = iteration_dir / selected_img
                    shutil.copy2(original_image_path, original_copy_path)
                    print(f"原始图像已复制到: {original_copy_path}")
                    
                    print(f"\n=== 标注说明 ===")
                    print(f"1. 下载以下文件到本地:")
                    print(f"   - 原始图像: {original_copy_path}")
                    print(f"   - 预测mask: {mask_path}")
                    print(f"2. 使用ImageJ或其他图像编辑软件修改预测mask")
                    print(f"3. 将修改后的mask上传到: {project_root / cfg_al.LABELED_DIR}")
                    print(f"4. 将原始图像也移动到: {project_root / cfg_al.LABELED_DIR}")
                    print(f"5. 确保图像和mask文件名对应")
        else:
            print(f"未找到图像 {selected_img} 的预测结果")
        
        # 显示当前标注进度
        current_labeled_count = len([f for f in os.listdir(project_root / cfg_al.LABELED_DIR) if f.endswith('.png')])
        print(f"当前已标注图像数量: {current_labeled_count}")
        print(f"本轮迭代: {iter_num}/{cfg_al.NUM_ITERATIONS}")
        
        # 等待用户确认标注完成
        input("完成mask标注后按 Enter 继续...")

        # 步骤4: 模型微调
        print("--- 开始微调模型 ---")
        train_dataset = SemSegmentationDataset(
            json_file_identifier=cfg_al.TRAIN_JSON_NAME,
            project_root=project_root,
            split='train'
        )
        train_loader = DataLoader(train_dataset, batch_size=cfg_al.BATCH_SIZE, shuffle=True)

        val_dataset = SemSegmentationDataset(
            json_file_identifier=cfg_al.VAL_JSON_NAME,
            project_root=project_root,
            split='val',
            return_dataset_name=True
        )
        val_loader = DataLoader(val_dataset, batch_size=cfg_al.BATCH_SIZE, shuffle=False)

        criterion = get_loss_function(cfg_al.LOSS_FUNCTION)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg_al.LEARNING_RATE, weight_decay=cfg_al.WEIGHT_DECAY)

        # 断点续训：尝试加载检查点
        start_epoch = 1
        best_val_dice = 0.0
        best_train_loss = float('inf')
        if cfg_al.RESUME_FROM_CHECKPOINT and os.path.exists(cfg_al.RESUMABLE_CHECKPOINT_PATH):
            start_epoch, best_train_loss, best_val_dice = load_checkpoint(model, optimizer, cfg_al.RESUMABLE_CHECKPOINT_PATH, device)
            start_epoch += 1  # 从下一个epoch开始

        best_val_dice = 0.0
        for epoch in range(1, cfg_al.NUM_EPOCHS_PER_ITER + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            eval_results = evaluate_model(model, val_loader, device, cfg_base.NUM_CLASSES)
            mean_dice = eval_results['mean_dice_scores'].mean()
            
            # 日志记录
            log_dict = {
                "iter": iter_num,
                "epoch": epoch, 
                "train_loss": train_loss, 
                "val_dice": mean_dice
            }
            # 记录每个class整体dice
            for class_name, dice in zip(eval_results['class_names'], eval_results['mean_dice_scores']):
                log_dict[f"val/all/{class_name}_dice"] = dice
            # 记录每个dataset的各类别dice
            for dataset_name, dataset_scores in eval_results.get('dataset_scores', {}).items():
                for class_name, dice in zip(eval_results['class_names'], dataset_scores):
                    log_dict[f"val/{dataset_name}/{class_name}_dice"] = dice
            
            logger.log(log_dict, step=epoch)
            print(f"Iter {iter_num}, Epoch {epoch} - Val Dice: {mean_dice:.4f}")

            # 保存检查点
            save_checkpoint(model, optimizer, epoch, train_loss, mean_dice, cfg_al.RESUMABLE_CHECKPOINT_PATH)

            # 保存最佳模型
            if mean_dice > best_val_dice:
                best_val_dice = mean_dice
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_dice': best_val_dice,
                    'train_loss': train_loss,
                    'config': {
                        'encoder_name': cfg_al.ENCODER_NAME,
                        'decoder_name': cfg_al.DECODER_NAME,
                        'n_channels': cfg_base.INPUT_DEPTH,
                        'n_classes': cfg_base.NUM_CLASSES
                    }
                }, cfg_al.BEST_MODEL_CHECKPOINT_PATH)
                print(f"新的最佳模型已保存! Val Dice: {best_val_dice:.4f}")

            # 每隔指定epoch数才输出详细的dice
            if epoch == 1 or epoch % cfg_al.DICE_EVAL_EPOCH_INTERVAL == 0:
                from utils.training_util import pretty_print_metrics
                pretty_print_metrics(eval_results)

        print(f"迭代 {iter_num} 完成，最佳验证Dice: {best_val_dice:.4f}")
        
        # 更新性能历史
        performance_history.append(best_val_dice)
        if best_val_dice > best_overall_dice:
            best_overall_dice = best_val_dice
            print(f"新的最佳整体性能! Dice: {best_overall_dice:.4f}")
        
        # 显示性能趋势
        if len(performance_history) > 1:
            improvement = best_val_dice - performance_history[-2]
            print(f"相比上一轮改进: {improvement:+.4f}")
        
        # 显示当前标注进度
        current_labeled_count = len([f for f in os.listdir(project_root / cfg_al.LABELED_DIR) if f.endswith('.png')])
        remaining_unlabeled = len([f for f in os.listdir(project_root / cfg_al.UNLABELED_POOL_DIR) if f.endswith('.png')])
        print(f"当前已标注: {current_labeled_count} 张，剩余无标签: {remaining_unlabeled} 张")

    print("\n--- 主动学习任务完成 ---")
    print(f"最终最佳性能: {best_overall_dice:.4f}")
    print(f"性能历史: {[f'{d:.4f}' for d in performance_history]}")
    logger.close()

if __name__ == '__main__':
    main_active_learning() 