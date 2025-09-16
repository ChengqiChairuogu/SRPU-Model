import torch
import torch.nn.functional as F
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
        
        # 处理MAPPING字典，将元组键转换为字符串键以避免easydict的类型错误
        raw_mapping = getattr(module, 'MAPPING', {})
        if raw_mapping:
            # 将元组键转换为字符串表示
            converted_mapping = {}
            original_mapping = {}
            for rgb_tuple, class_id in raw_mapping.items():
                if isinstance(rgb_tuple, tuple):
                    # 将RGB元组转换为字符串键，例如 "(0, 0, 255)" -> "0_0_255"
                    rgb_key = "_".join(map(str, rgb_tuple))
                    # 同时保存转换后的原始映射，避免easydict的元组键问题
                    original_mapping[rgb_key] = class_id
                else:
                    rgb_key = str(rgb_tuple)
                    original_mapping[rgb_key] = class_id
                converted_mapping[rgb_key] = class_id
            base_config.MAPPING = converted_mapping
            # 保存转换后的原始映射用于向后兼容
            base_config.ORIGINAL_MAPPING = original_mapping
        else:
            base_config.MAPPING = {}
            base_config.ORIGINAL_MAPPING = {}
        
        # 处理CLASS_NAMES字典，将整数键转换为字符串键以避免easydict的类型错误
        raw_class_names = getattr(module, 'CLASS_NAMES', {})
        if raw_class_names and isinstance(raw_class_names, dict):
            # 将整数键转换为字符串键
            converted_class_names = {}
            for class_id, class_name in raw_class_names.items():
                converted_class_names[str(class_id)] = class_name
            base_config.CLASS_NAMES = converted_class_names
        else:
            # 如果没有CLASS_NAMES字典，使用默认的列表格式
            base_config.CLASS_NAMES = ['class_%d' % i for i in range(base_config.NUM_CLASSES)]
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
        # 从转换后的映射重建RGB元组到类别ID的映射
        mapping = {}
        for rgb_key, class_id in base_config.MAPPING.items():
            if '_' in rgb_key:
                # 将字符串键转换回RGB元组
                rgb_values = tuple(map(int, rgb_key.split('_')))
                mapping[rgb_values] = class_id
            else:
                # 处理非元组键的情况
                mapping[rgb_key] = class_id
        
        out = np.zeros(mask.shape[:2], dtype=np.int64)
        for rgb, idx in mapping.items():
            if isinstance(rgb, tuple):
                out[(mask == rgb).all(axis=-1)] = idx
            else:
                # 处理非元组键的情况
                out[(mask == rgb)] = idx
        return out
    return mask

def dice_score(pred, gt, num_classes):
    """计算Dice系数，与训练时的计算方式保持一致"""
    scores = []
    for i in range(num_classes):
        pred_i = (pred == i)
        gt_i = (gt == i)
        intersection = (pred_i & gt_i).sum()
        union = pred_i.sum() + gt_i.sum()
        if union == 0:
            scores.append(1.0)
        else:
            # 使用与训练时一致的平滑因子
            scores.append(2.0 * intersection / (union + 1e-6))
    return np.array(scores)

def iou_score(pred, gt, num_classes):
    """计算IoU系数，与训练时的计算方式保持一致"""
    scores = []
    for i in range(num_classes):
        pred_i = (pred == i)
        gt_i = (gt == i)
        intersection = (pred_i & gt_i).sum()
        union = (pred_i | gt_i).sum()
        if union == 0:
            scores.append(1.0)
        else:
            # 使用与训练时一致的平滑因子
            scores.append(intersection / (union + 1e-6))
    return np.array(scores)

def calculate_psnr_ssim(pred_mask, gt_mask, num_classes):
    """计算PSNR和SSIM指标（如果适用）"""
    try:
        from skimage.metrics import peak_signal_noise_ratio as psnr
        from skimage.metrics import structural_similarity as ssim
        
        psnr_scores = []
        ssim_scores = []
        
        for i in range(num_classes):
            pred_i = (pred_mask == i).astype(np.float32)
            gt_i = (gt_mask == i).astype(np.float32)
            
            # 计算PSNR
            try:
                data_range = gt_i.max() - gt_i.min()
                if data_range == 0:
                    data_range = 1.0
                psnr_val = psnr(gt_i, pred_i, data_range=data_range)
                if isinstance(psnr_val, tuple):
                    psnr_val = psnr_val[0]
                psnr_scores.append(float(psnr_val))
            except Exception as e:
                psnr_scores.append(0.0)
            
            # 计算SSIM
            try:
                data_range = gt_i.max() - gt_i.min()
                if data_range == 0:
                    data_range = 1.0
                ssim_val = ssim(gt_i, pred_i, data_range=data_range)
                if isinstance(ssim_val, tuple):
                    ssim_val = ssim_val[0]
                ssim_scores.append(float(ssim_val))
            except Exception as e:
                ssim_scores.append(0.0)
        
        return np.array(psnr_scores), np.array(ssim_scores)
    except ImportError:
        # 如果没有安装skimage，返回零值
        return np.zeros(num_classes), np.zeros(num_classes)

# 核心函数
def evaluate_model(config, base_config):
    """评估模型性能"""
    print("--- 开始模型评估任务 ---")
    
    if config.EVALUATION_MODE == 'random_sample':
        print("--- 随机抽取评估模式 ---")
        
        # 检查设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"设备检测: {'CUDA (GPU)' if device.type == 'cuda' else 'CPU'} 将被使用。")
        
        # 创建输出目录
        output_dir = Path(config.OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"结果将保存至: {output_dir}")
        
        # 定义要评估的数据集
        # 注意：只有master文件包含完整的datasets_info，其他单独的文件需要特殊处理
        datasets_to_evaluate = [
            ('dataset1_LInCl', 'json/dataset1_LInCl.json'),
            ('dataset2_LPSCl', 'json/dataset2_LPSCl.json'),
            ('dataset3_LNOCl', 'json/dataset3_LNOCl.json'),
            ('master', 'json/master_sharpness_averaged_dataset.json')
        ]
        
        all_metrics = {}
        
        # 评估每个数据集
        for dataset_name, json_file in datasets_to_evaluate:
            print(f"\n{'='*20} 评估数据集: {dataset_name} {'='*20}")
            
            # 检查数据集一致性
            check_dataset_consistency(dataset_name, json_file)
            
            # 临时修改配置以使用当前数据集
            original_json = config.DATA_LOADER_CONFIG.val.json_file_identifier
            config.DATA_LOADER_CONFIG.val.json_file_identifier = json_file
            
            try:
                # 评估验证集
                print(f"\n--- 评估 {dataset_name} 验证集 ===")
                val_metrics = evaluate_random_samples(config, base_config, split='val')
                
                # 如果配置了同时评估训练集
                if hasattr(config, 'EVALUATE_BOTH_SPLITS') and config.EVALUATE_BOTH_SPLITS:
                    print(f"\n--- 评估 {dataset_name} 训练集 ===")
                    train_metrics = evaluate_random_samples(config, base_config, split='train')
                    
                    # 合并结果
                    all_metrics[dataset_name] = {
                        'val': val_metrics,
                        'train': train_metrics
                    }
                    
                    print(f"\n=== {dataset_name} 综合评估结果 ===")
                    print("验证集性能:")
                    print_metrics_summary(val_metrics)
                    print("\n训练集性能:")
                    print_metrics_summary(train_metrics)
                else:
                    all_metrics[dataset_name] = val_metrics
                    print(f"\n=== {dataset_name} 验证集评估结果 ===")
                    print_metrics_summary(val_metrics)
                    
            except Exception as e:
                print(f"评估数据集 {dataset_name} 时出错: {e}")
                continue
            finally:
                # 恢复原始配置
                config.DATA_LOADER_CONFIG.val.json_file_identifier = original_json
        
        # 显示所有数据集的综合对比
        if len(all_metrics) > 1:
            print("\n" + "="*60)
            print("=== 所有数据集性能对比 ===")
            print("="*60)
            print_overall_metrics(all_metrics)
            
            # 分析训练和评估结果的差异
            analyze_training_evaluation_discrepancy()
    
    elif config.EVALUATION_MODE == 'inference_result':
        print("--- 推理结果评估模式 ---")
        evaluate_inference_results(config, base_config)
    
    else:
        raise ValueError(f"不支持的评估模式: {config.EVALUATION_MODE}")
    
    print("--- 模型评估任务完成 ---")

def evaluate_random_samples(config, base_config, split='val'):
    """模式1: 随机抽取数据集中的图片进行评估"""
    print(f"--- 随机抽取 {split} 集评估模式 ---")
    
    device = get_device()
    output_dir = config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    print(f"结果将保存至: {output_dir}")
    
    print(f"加载 {split} 数据集...")
    
    # 检查JSON文件是否包含datasets_info，如果没有则添加默认路径
    json_path = Path(__file__).resolve().parent.parent / config.DATA_LOADER_CONFIG.val.json_file_identifier
    if json_path.exists():
        with open(json_path, 'r') as f:
            data_info = json.load(f)
        
        # 如果缺少datasets_info，添加默认路径
        if 'datasets_info' not in data_info:
            print("检测到缺少datasets_info字段，添加默认路径...")
            # 从JSON文件名推断数据集名称
            json_filename = Path(config.DATA_LOADER_CONFIG.val.json_file_identifier).stem
            if 'dataset1_LInCl' in json_filename:
                data_info['datasets_info'] = {
                    'dataset1_LInCl': {
                        'raw_image_root': 'datasets/dataset1_LInCl/raw_images',
                        'mask_root': 'datasets/dataset1_LInCl/masks_3class'
                    }
                }
            elif 'dataset2_LPSCl' in json_filename:
                data_info['datasets_info'] = {
                    'dataset2_LPSCl': {
                        'raw_image_root': 'datasets/dataset2_LPSCl/raw_images',
                        'mask_root': 'datasets/dataset2_LPSCl/masks_3class'
                    }
                }
            elif 'dataset3_LNOCl' in json_filename:
                data_info['datasets_info'] = {
                    'dataset3_LNOCl': {
                        'raw_image_root': 'datasets/dataset3_LNOCl/raw_images',
                        'mask_root': 'datasets/dataset3_LNOCl/masks_3class'
                    }
                }
            
            # 将修改后的数据写回JSON文件（临时）
            temp_json_path = json_path.parent / f"temp_{json_path.name}"
            with open(temp_json_path, 'w') as f:
                json.dump(data_info, f, indent=2)
            
            # 临时使用修改后的JSON文件
            temp_json_identifier = f"temp_{Path(config.DATA_LOADER_CONFIG.val.json_file_identifier).name}"
            val_dataset = SemSegmentationDataset(
                json_file_identifier=temp_json_identifier,
                project_root=Path(__file__).resolve().parent.parent,
                split=split
            )
            
            # 清理临时文件
            temp_json_path.unlink()
        else:
            val_dataset = SemSegmentationDataset(
                json_file_identifier=config.DATA_LOADER_CONFIG.val.json_file_identifier,
                project_root=Path(__file__).resolve().parent.parent,
                split=split
            )
    else:
        val_dataset = SemSegmentationDataset(
            json_file_identifier=config.DATA_LOADER_CONFIG.val.json_file_identifier,
            project_root=Path(__file__).resolve().parent.parent,
            split=split
        )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=config.DATA_LOADER_CONFIG.val.num_workers)
    
    print("加载模型...")
    
    # 创建监督学习模型
    from models.segmentation_unet import SegmentationUNet
    model = SegmentationUNet(
        encoder_name=config.MODEL_CONFIG.encoder_name,
        decoder_name=config.MODEL_CONFIG.decoder_name,
        n_channels=base_config.INPUT_DEPTH,
        n_classes=base_config.NUM_CLASSES
    )
    print(f"创建监督学习模型: {config.MODEL_CONFIG.model_class}")
    
    checkpoint_path = config.CHECKPOINT_PATH
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到模型权重文件: {checkpoint_path}")
    
    print(f"从 {checkpoint_path} 加载权重...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 根据模型类型处理权重加载
    if config.MODEL_TYPE == 'supervised':
        # 监督学习模型权重加载
        if 'model_state_dict' in checkpoint:
            # 这是一个完整的训练检查点，需要提取模型权重
            state_dict = checkpoint['model_state_dict']
            print(f"从检查点加载模型权重，训练轮次: {checkpoint.get('epoch', 'unknown')}")
        else:
            # 这是直接的模型权重文件
            state_dict = checkpoint
            print("直接加载模型权重文件")
        
        model.load_state_dict(state_dict)
        
    # 确保模型在正确的设备上并设置为评估模式
    model = model.to(device)
    model.eval()
    
    num_to_inspect = config.NUM_IMAGES_TO_INSPECT if config.NUM_IMAGES_TO_INSPECT > 0 else len(val_loader)
    
    dice_all = []
    iou_all = []
    lines = []
    
    with torch.no_grad():
        for i, (images, true_masks) in enumerate(tqdm(val_loader, total=num_to_inspect, desc="评估进度")):
            if i >= num_to_inspect: break
            
            images = images.to(device, dtype=torch.float32)
            
            # 监督学习模型：分割任务
            pred_logits = model(images)
            # 使用与训练时一致的预测方式：argmax而不是阈值化
            pred_masks = torch.argmax(pred_logits, dim=1)
            
            image_np = images.cpu().numpy().squeeze()
            true_mask_np = true_masks.cpu().numpy().squeeze()
            pred_mask_np = pred_masks.cpu().numpy().squeeze()

            
            # 确保所有数据都是正确的维度用于可视化
            # 图像应该是 (height, width) 或 (height, width, channels)
            # 掩码应该是 (height, width)
            
            # 处理图像数据
            if image_np.ndim == 3:
                if image_np.shape[0] == 3:  # (channels, height, width)
                    image_np = image_np.transpose(1, 2, 0)  # 转换为 (height, width, channels)
                elif image_np.shape[2] == 3:  # (height, width, channels)
                    pass  # 已经是正确的格式
                else:
                    # 取中间通道
                    s = image_np.shape[0] // 2
                    image_np = image_np[s]
            
            # 处理真实掩码数据
            if true_mask_np.ndim == 3:
                if true_mask_np.shape[0] == base_config.NUM_CLASSES:  # (classes, height, width)
                    # 取第一个类别作为主要掩码
                    true_mask_np = true_mask_np[0]
                elif true_mask_np.shape[2] == base_config.NUM_CLASSES:  # (height, width, classes)
                    # 取第一个类别作为主要掩码
                    true_mask_np = true_mask_np[:, :, 0]
                else:
                    # 取中间通道
                    s = true_mask_np.shape[0] // 2
                    true_mask_np = true_mask_np[s]
            
            # 处理预测掩码数据
            if pred_mask_np.ndim == 3:
                if pred_mask_np.shape[0] == base_config.NUM_CLASSES:  # (classes, height, width)
                    # 取第一个类别作为主要掩码
                    pred_mask_np = pred_mask_np[0]
                elif pred_mask_np.shape[2] == base_config.NUM_CLASSES:  # (height, width, classes)
                    # 取第一个类别作为主要掩码
                    pred_mask_np = pred_mask_np[:, :, 0]
                else:
                    # 取中间通道
                    s = pred_mask_np.shape[0] // 2
                    pred_mask_np = pred_mask_np[s]
            
            # 可视化
            if config.VISUALIZATION.save_images:
                fig, axes = plt.subplots(1, 3, figsize=config.VISUALIZATION.figure_size)
                fig.suptitle(f'Sample {i+1}', fontsize=16)
                axes[0].imshow(image_np, cmap='gray'); axes[0].set_title('Original Image'); axes[0].axis('off')
                axes[1].imshow(true_mask_np, cmap='gray'); axes[1].set_title('Ground Truth Mask'); axes[1].axis('off')
                axes[2].imshow(pred_mask_np, cmap='gray'); axes[2].set_title('Model Prediction'); axes[2].axis('off')
                plt.savefig(os.path.join(output_dir, f'evaluation_sample_{i+1:04d}.{config.VISUALIZATION.image_format}'), 
                           bbox_inches='tight', dpi=config.VISUALIZATION.dpi)
                plt.close(fig)
            
            # 指标计算 - 与训练时的评估方式保持一致
            if pred_mask_np.shape != true_mask_np.shape:
                pred_mask_np = np.array(Image.fromarray(pred_mask_np.astype(np.uint8)).resize(true_mask_np.shape[::-1], resample=Image.NEAREST))
            
            # 确保数据类型一致
            pred_mask_np = pred_mask_np.astype(np.int64)
            true_mask_np = true_mask_np.astype(np.int64)
            
            # 计算主要指标
            dice = dice_score(pred_mask_np, true_mask_np, base_config.NUM_CLASSES)
            iou = iou_score(pred_mask_np, true_mask_np, base_config.NUM_CLASSES)
            
            # 计算PSNR和SSIM（如果适用）
            psnr_scores, ssim_scores = calculate_psnr_ssim(pred_mask_np, true_mask_np, base_config.NUM_CLASSES)
            
            dice_all.append(dice)
            iou_all.append(iou)
            
            # 打印详细的每样本指标
            line = f"Sample {i+1}: "
            for j in range(base_config.NUM_CLASSES):
                cname = base_config.CLASS_NAMES.get(str(j), f'class_{j}')
                line += f" {cname} Dice: {dice[j]:.4f} IoU: {iou[j]:.4f}"
                if psnr_scores[j] > 0:  # 只在有效时显示PSNR/SSIM
                    line += f" PSNR: {psnr_scores[j]:.2f}dB SSIM: {ssim_scores[j]:.4f}"
            print(line)
            lines.append(line)
            
            # 添加调试信息：显示每个类别的像素统计
            for j in range(base_config.NUM_CLASSES):
                cname = base_config.CLASS_NAMES.get(str(j), f'class_{j}')
                pred_pixels = (pred_mask_np == j).sum()
                gt_pixels = (true_mask_np == j).sum()
                intersection_pixels = ((pred_mask_np == j) & (true_mask_np == j)).sum()
                
                # 计算每个类别的Dice和IoU
                if gt_pixels + pred_pixels > 0:
                    class_dice = 2.0 * intersection_pixels / (gt_pixels + pred_pixels + 1e-6)
                else:
                    class_dice = 1.0 if gt_pixels == 0 and pred_pixels == 0 else 0.0
                
                if gt_pixels + pred_pixels - intersection_pixels > 0:
                    class_iou = intersection_pixels / (gt_pixels + pred_pixels - intersection_pixels + 1e-6)
                else:
                    class_iou = 1.0 if gt_pixels == 0 and pred_pixels == 0 else 0.0
                
                print(f"    {cname}: 预测像素={pred_pixels}, GT像素={gt_pixels}, 交集={intersection_pixels}")
                print(f"         {cname} Dice: {class_dice:.4f}, IoU: {class_iou:.4f}")
            
            # 添加样本级别的详细分析
            print(f"    样本 {i+1} 总体统计:")
            print(f"      图像形状: {image_np.shape}, 数据类型: {image_np.dtype}")
            print(f"      GT掩码形状: {true_mask_np.shape}, 数据类型: {true_mask_np.dtype}")
            print(f"      预测掩码形状: {pred_mask_np.shape}, 数据类型: {pred_mask_np.dtype}")
            print(f"      GT掩码值范围: [{true_mask_np.min()}, {true_mask_np.max()}]")
            print(f"      预测掩码值范围: [{pred_mask_np.min()}, {pred_mask_np.max()}]")
            print(f"      GT掩码唯一值: {np.unique(true_mask_np)}")
            print(f"      预测掩码唯一值: {np.unique(pred_mask_np)}")
    
    # 根据模型类型汇总不同的指标
    if config.MODEL_TYPE == 'supervised':
        # 监督学习模型：汇总分割指标
        dice_all = np.array(dice_all)
        iou_all = np.array(iou_all)
        
        # 计算每个类别的平均指标
        lines.append("\n=== 总体均值 ===")
        print("\n=== 总体均值 ===")
        
        for j in range(base_config.NUM_CLASSES):
            cname = base_config.CLASS_NAMES.get(str(j), f'class_{j}')
            dice_mean = dice_all[:,j].mean()
            iou_mean = iou_all[:,j].mean()
            
            # 计算标准差
            dice_std = dice_all[:,j].std()
            iou_std = iou_all[:,j].std()
            
            s = f"{cname} Dice: {dice_mean:.4f}±{dice_std:.4f} IoU: {iou_mean:.4f}±{iou_std:.4f}"
            print(s)
            lines.append(s)
        
        # 计算总体平均
        overall_dice = dice_all.mean()
        overall_iou = iou_all.mean()
        overall_dice_std = dice_all.std()
        overall_iou_std = iou_all.std()
        
        print(f"\n=== 综合指标 ===")
        print(f"平均Dice: {overall_dice:.4f}±{overall_dice_std:.4f}")
        print(f"平均IoU: {overall_iou:.4f}±{overall_iou_std:.4f}")
        
        lines.append(f"\n=== 综合指标 ===")
        lines.append(f"平均Dice: {overall_dice:.4f}±{overall_dice_std:.4f}")
        lines.append(f"平均IoU: {overall_iou:.4f}±{overall_iou_std:.4f}")
        
        # 返回详细的指标信息
        metrics_summary = {
            'Dice': dice_all,
            'IoU': iou_all,
            'ClassNames': base_config.CLASS_NAMES,
            'OverallDice': overall_dice,
            'OverallIoU': overall_iou,
            'DiceStd': overall_dice_std,
            'IoUStd': overall_iou_std
        }
        

    
    # 保存结果
    result_save_path = config.RESULT_SAVE_PATH or os.path.join(output_dir, 'evaluation_results.txt')
    
    # 确保结果文件的目录存在
    result_dir = os.path.dirname(result_save_path)
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)
    
    with open(result_save_path, 'w', encoding='utf-8') as f:
        for l in lines:
            f.write(l + '\n')
    print(f"评估结果已保存到: {result_save_path}")
    
    print(f"\n评估完成！生成 {num_to_inspect} 张对比图和指标报告。")
    return metrics_summary

def evaluate_inference_results(config, base_config):
    """模式2: 评估推理预测出的图像"""
    print("--- 推理结果评估模式 ---")
    
    output_dir = config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    print(f"结果将保存至: {output_dir}")
    
    # 获取推理结果文件列表
    pred_mask_dir = Path(config.PRED_MASK_DIR)
    if not pred_mask_dir.exists():
        raise FileNotFoundError(f"推理结果目录不存在: {pred_mask_dir}")
    
    pred_files = list(pred_mask_dir.glob(f"*{config.PRED_MASK_EXTENSION}"))
    if not pred_files:
        raise FileNotFoundError(f"在 {pred_mask_dir} 中未找到推理结果文件")
    
    print(f"找到 {len(pred_files)} 个推理结果文件")
    
    # 限制评估数量
    if config.MAX_EVALUATION_IMAGES > 0:
        pred_files = pred_files[:config.MAX_EVALUATION_IMAGES]
        print(f"将评估前 {len(pred_files)} 个文件")
    
    dice_all = []
    iou_all = []
    lines = []
    evaluated_count = 0
    
    for pred_file in tqdm(pred_files, desc="评估推理结果"):
        try:
            # 加载预测掩码
            pred_mask = np.array(Image.open(pred_file))
            
            # 查找对应的GT掩码
            gt_mask = find_gt_mask(pred_file, config, base_config)
            if gt_mask is None:
                if config.ONLY_EVALUATE_WITH_GT:
                    print(f"跳过 {pred_file.name}: 未找到对应GT")
                    continue
                else:
                    print(f"警告: {pred_file.name} 未找到对应GT，将跳过指标计算")
                    continue
            
            # 指标计算
            dice = dice_score(pred_mask, gt_mask, base_config.NUM_CLASSES)
            iou = iou_score(pred_mask, gt_mask, base_config.NUM_CLASSES)
            dice_all.append(dice)
            iou_all.append(dice)
            
            # 可视化
            if config.VISUALIZATION.save_images:
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                fig.suptitle(f'Inference Result: {pred_file.stem}', fontsize=16)
                axes[0].imshow(pred_mask, cmap='gray'); axes[0].set_title('Predicted Mask'); axes[0].axis('off')
                axes[1].imshow(gt_mask, cmap='gray'); axes[1].set_title('Ground Truth'); axes[1].axis('off')
                plt.savefig(os.path.join(output_dir, f'inference_eval_{pred_file.stem}.{config.VISUALIZATION.image_format}'), 
                           bbox_inches='tight', dpi=config.VISUALIZATION.dpi)
                plt.close(fig)
            
            line = f"{pred_file.name}: "
            for j in range(base_config.NUM_CLASSES):
                cname = base_config.CLASS_NAMES.get(str(j), f'class_{j}')
                line += f" {cname} Dice: {dice[j]:.4f} IoU: {iou[j]:.4f}"
            print(line)
            lines.append(line)
            
            evaluated_count += 1
            
        except Exception as e:
            print(f"处理 {pred_file.name} 时出错: {e}")
            continue
    
    if evaluated_count == 0:
        print("没有成功评估任何文件！")
        return
    
    # 汇总指标
    dice_all = np.array(dice_all)
    iou_all = np.array(iou_all)
    lines.append("\n=== 总体均值 ===")
    for j in range(base_config.NUM_CLASSES):
        cname = base_config.CLASS_NAMES.get(str(j), f'class_{j}')
        s = f"{cname} Dice: {dice_all[:,j].mean():.4f} IoU: {iou_all[:,j].mean():.4f}"
        print(s)
        lines.append(s)
    print(f"平均Dice: {dice_all.mean():.4f}")
    print(f"平均IoU: {iou_all.mean():.4f}")
    lines.append(f"平均Dice: {dice_all.mean():.4f}")
    lines.append(f"平均IoU: {iou_all.mean():.4f}")
    
    # 保存结果
    result_save_path = config.RESULT_SAVE_PATH or os.path.join(output_dir, 'inference_evaluation_results.txt')
    with open(result_save_path, 'w', encoding='utf-8') as f:
        for l in lines:
            f.write(l + '\n')
    print(f"评估结果已保存到: {result_save_path}")
    
    print(f"\n推理结果评估完成！成功评估 {evaluated_count} 个文件。")

def find_gt_mask(pred_file, config, base_config):
    """查找对应的GT掩码"""
    # 首先尝试从GT目录直接查找
    if hasattr(config, 'GT_MASK_DIR') and config.GT_MASK_DIR:
        gt_dir = Path(config.GT_MASK_DIR)
        # 尝试不同的GT文件名模式
        gt_patterns = [
            pred_file.stem.replace('_mask', '') + '.png',  # 移除_mask后缀
            pred_file.stem.replace('_mask', '') + '_mask.png',  # 保持_mask后缀
            pred_file.stem + '.png',  # 完全匹配
        ]
        
        for pattern in gt_patterns:
            gt_file = gt_dir / pattern
            if gt_file.exists():
                return np.array(Image.open(gt_file))
    
    # 从JSON文件查找
    if hasattr(config, 'GT_JSON_PATHS') and config.GT_JSON_PATHS:
        # 这里可以实现从JSON文件查找GT的逻辑
        # 由于实现较复杂，暂时返回None
        pass
    
    return None

def print_metrics_summary(metrics_dict):
    """打印单个数据集的指标摘要"""
    print(f"\n--- 指标摘要 ---")
    

    
    # 监督学习模型：显示分割指标
    if 'ClassNames' in metrics_dict:
        for j in range(len(metrics_dict['ClassNames'])):
            cname = metrics_dict['ClassNames'].get(str(j), f'class_{j}')
            dice_mean = metrics_dict['Dice'][:, j].mean()
            iou_mean = metrics_dict['IoU'][:, j].mean()
            dice_std = metrics_dict['Dice'][:, j].std()
            iou_std = metrics_dict['IoU'][:, j].std()
            s = f"{cname} Dice: {dice_mean:.4f}±{dice_std:.4f} IoU: {iou_mean:.4f}±{iou_std:.4f}"
            print(s)
        
        # 使用新的指标结构（如果可用）
        if 'OverallDice' in metrics_dict:
            overall_dice = metrics_dict['OverallDice']
            overall_iou = metrics_dict['OverallIoU']
            overall_dice_std = metrics_dict.get('DiceStd', 0.0)
            overall_iou_std = metrics_dict.get('IoUStd', 0.0)
            print(f"平均Dice: {overall_dice:.4f}±{overall_dice_std:.4f}")
            print(f"平均IoU: {overall_iou:.4f}±{overall_iou_std:.4f}")
        else:
            # 向后兼容
            print(f"平均Dice: {metrics_dict['Dice'].mean():.4f}")
            print(f"平均IoU: {metrics_dict['IoU'].mean():.4f}")
    else:
        print("警告: 无法识别的指标结构")

def analyze_training_evaluation_discrepancy():
    """分析训练和评估结果的差异"""
    print("\n" + "="*80)
    print("=== 训练 vs 评估结果差异分析 ===")
    print("="*80)
    
    print("\n🔍 **可能的原因分析：**")
    print("1. **数据分布差异**：训练时的验证集与评估时的验证集可能不完全一致")
    print("2. **数据预处理差异**：训练时可能有数据增强，评估时没有")
    print("3. **模型状态差异**：训练时模型可能处于不同状态（如dropout激活）")
    print("4. **类别不平衡**：某些类别样本较少，导致评估不稳定")
    print("5. **样本选择差异**：评估时可能选择了更容易预测的样本")
    
    print("\n📊 **建议的调试步骤：**")
    print("1. 检查训练和评估时使用的验证集是否完全一致")
    print("2. 比较训练和评估时的数据预处理流程")
    print("3. 分析每个类别的样本数量和分布")
    print("4. 检查模型在训练和评估时的状态设置")
    print("5. 对比训练日志中的详细指标")
    
    print("\n" + "="*80)

def check_dataset_consistency(dataset_name, json_file):
    """检查数据集的一致性"""
    print(f"\n🔍 检查数据集 {dataset_name} 的一致性...")
    
    try:
        json_path = Path(__file__).resolve().parent.parent / json_file
        if json_path.exists():
            with open(json_path, 'r') as f:
                data_info = json.load(f)
            
            # 检查样本数量
            total_samples = len(data_info.get('samples', []))
            val_samples = len([s for s in data_info.get('samples', []) if s.get('split') == 'val'])
            train_samples = len([s for s in data_info.get('samples', []) if s.get('split') == 'train'])
            
            print(f"  总样本数: {total_samples}")
            print(f"  验证集样本数: {val_samples}")
            print(f"  训练集样本数: {train_samples}")
            
            # 检查类别分布
            if 'samples' in data_info:
                class_counts = {}
                for sample in data_info['samples']:
                    dataset = sample.get('dataset', 'unknown')
                    if dataset not in class_counts:
                        class_counts[dataset] = 0
                    class_counts[dataset] += 1
                
                print(f"  数据集分布: {class_counts}")
            
            return True
        else:
            print(f"  ❌ JSON文件不存在: {json_file}")
            return False
    except Exception as e:
        print(f"  ❌ 检查数据集一致性时出错: {e}")
        return False

def print_overall_metrics(all_metrics):
    """打印综合评估结果"""
    print("\n--- 综合评估结果 ===")
    
    # 检查是否是多个数据集的对比
    if any('val' in metrics and 'train' in metrics for metrics in all_metrics.values()):
        # 这是多个数据集的对比
        print("=== 多数据集性能对比 ===")
        
        # 创建对比表格
        print(f"{'数据集':<20} {'验证集Dice':<15} {'验证集IoU':<15} {'训练集Dice':<15} {'训练集IoU':<15}")
        print("-" * 80)
        
        for dataset_name, metrics in all_metrics.items():
            if 'val' in metrics and 'train' in metrics:
                val_dice = metrics['val'].get('OverallDice', metrics['val']['Dice'].mean())
                val_iou = metrics['val'].get('OverallIoU', metrics['val']['IoU'].mean())
                train_dice = metrics['train'].get('OverallDice', metrics['train']['Dice'].mean())
                train_iou = metrics['train'].get('OverallIoU', metrics['train']['IoU'].mean())
                
                print(f"{dataset_name:<20} {val_dice:<15.4f} {val_iou:<15.4f} {train_dice:<15.4f} {train_iou:<15.4f}")
        
        # 计算所有数据集的总体平均
        print("-" * 80)
        total_val_dice = sum(metrics['val'].get('OverallDice', metrics['val']['Dice'].mean()) for metrics in all_metrics.values() if 'val' in metrics)
        total_val_iou = sum(metrics['val'].get('OverallIoU', metrics['val']['IoU'].mean()) for metrics in all_metrics.values() if 'val' in metrics)
        total_train_dice = sum(metrics['train'].get('OverallDice', metrics['train']['Dice'].mean()) for metrics in all_metrics.values() if 'train' in metrics)
        total_train_iou = sum(metrics['train'].get('OverallIoU', metrics['train']['IoU'].mean()) for metrics in all_metrics.values() if 'train' in metrics)
        
        num_datasets = len(all_metrics)
        avg_val_dice = total_val_dice / num_datasets
        avg_val_iou = total_val_iou / num_datasets
        avg_train_dice = total_train_dice / num_datasets
        avg_train_iou = total_train_iou / num_datasets
        
        print(f"{'总体平均':<20} {avg_val_dice:<15.4f} {avg_val_iou:<15.4f} {avg_train_dice:<15.4f} {avg_train_iou:<15.4f}")
        
    else:
        # 这是单个数据集的多个split对比
        print("=== 单数据集多Split对比 ===")
        
        # 计算总体平均
        total_dice = 0
        total_iou = 0
        total_samples = 0
        
        for split, metrics in all_metrics.items():
            print(f"\n{split} 集指标:")
            if 'ClassNames' in metrics:
                for j in range(len(metrics['ClassNames'])):
                    cname = metrics['ClassNames'].get(str(j), f'class_{j}')
                    dice_mean = metrics['Dice'][:,j].mean()
                    iou_mean = metrics['IoU'][:,j].mean()
                    dice_std = metrics['Dice'][:,j].std()
                    iou_std = metrics['IoU'][:,j].std()
                    print(f"  {cname}: Dice: {dice_mean:.4f}±{dice_std:.4f}, IoU: {iou_mean:.4f}±{iou_std:.4f}")
                
                # 使用新的指标结构（如果可用）
                if 'OverallDice' in metrics:
                    split_dice = metrics['OverallDice']
                    split_iou = metrics['OverallIoU']
                    split_dice_std = metrics.get('DiceStd', 0.0)
                    split_iou_std = metrics.get('IoUStd', 0.0)
                    print(f"  {split} 集平均: Dice: {split_dice:.4f}±{split_dice_std:.4f}, IoU: {split_iou:.4f}±{split_iou_std:.4f}")
                else:
                    split_dice = metrics['Dice'].mean()
                    split_iou = metrics['IoU'].mean()
                    print(f"  {split} 集平均: Dice: {split_dice:.4f}, IoU: {split_iou:.4f}")
                
                # 累加用于计算总体平均
                total_dice += split_dice
                total_iou += split_iou
                total_samples += 1
        
        # 计算总体平均
        if total_samples > 0:
            overall_dice = total_dice / total_samples
            overall_iou = total_iou / total_samples
            print(f"\n=== 总体平均性能 ===")
            print(f"averageDice: {overall_dice:.4f}")
            print(f"averageIoU: {overall_iou:.4f}")



# 主程序入口
if __name__ == '__main__':
    print("--- 正在初始化评估脚本 ---")
    base_config = build_base_config_from_module(base_config_module)
    evaluation_config = get_evaluation_config()
    evaluate_model(evaluation_config, base_config) 