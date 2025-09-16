import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import torchvision
from torchvision import models
from torchvision.transforms import Normalize
from skimage.metrics import structural_similarity as skimage_ssim

# 导入base配置
try:
    from configs import base as cfg_base
except ImportError:
    # 如果无法导入，使用默认值
    class MockCfgBase:
        CLASS_NAMES = {0: "carbon", 1: "SE", 2: "AM"}
    cfg_base = MockCfgBase()

def ensure_interpolate_size(target_shape):
    """
    确保interpolate的size参数格式正确
    Args:
        target_shape: 目标形状，可能是torch.Size([H, W])或torch.Size([H])等
    Returns:
        正确格式的size参数，确保是(H, W)格式
    """
    if len(target_shape) == 1:
        # 如果target_shape是单个数字，转换为(H, W)格式
        return (target_shape[0], target_shape[0])
    elif len(target_shape) == 2:
        # 如果已经是(H, W)格式，直接返回
        return target_shape
    else:
        # 如果长度大于2，取最后两个维度
        return target_shape[-2:]

# --- 手动实现的损失函数 ---
class DiceLoss(nn.Module):
    def __init__(self, include_background=True, to_onehot_y=False, softmax=False, other_act=None, squared_pred=False, jaccard=False):
        super().__init__()
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.softmax = softmax
        self.other_act = other_act
        self.squared_pred = squared_pred
        self.jaccard = jaccard

    def forward(self, input, target):
        if self.softmax:
            input = F.softmax(input, dim=1)
        if self.other_act:
            input = self.other_act(input)
        if self.to_onehot_y:
            target = F.one_hot(target, num_classes=input.shape[1]).permute(0, 3, 1, 2)
        
        if not self.include_background:
            input = input[:, 1:]
            target = target[:, 1:]

        assert input.shape == target.shape, "input and target dimensions must match"
        
        reduce_axis = list(range(2, len(input.shape)))
        intersection = torch.sum(input * target, dim=reduce_axis)
        
        if self.squared_pred:
            ground_o = torch.sum(input**2, dim=reduce_axis)
            ground_g = torch.sum(target**2, dim=reduce_axis)
        else:
            ground_o = torch.sum(input, dim=reduce_axis)
            ground_g = torch.sum(target, dim=reduce_axis)
            
        denominator = ground_o + ground_g
        
        if self.jaccard:
            denominator = 2.0 * denominator - intersection

        f = (2.0 * intersection) / (denominator + 1e-6)
        return 1.0 - f.mean()

class DiceCELoss(nn.Module):
    def __init__(self, lambda_dice=1.0, lambda_ce=1.0):
        super().__init__()
        self.dice_loss = DiceLoss(softmax=True, to_onehot_y=True)
        self.ce_loss = nn.CrossEntropyLoss()
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce

    def forward(self, pred, target):
        # 确保模型输出和标签的空间维度一致
        if pred.shape[-2:] != target.shape[-2:]:
            target_shape = ensure_interpolate_size(target.shape[-2:])
            pred = F.interpolate(pred, size=target_shape, mode='bilinear', align_corners=False)
        return self.lambda_dice * self.dice_loss(pred, target) + self.lambda_ce * self.ce_loss(pred, target)

class DiceBCELoss(nn.Module):
    def __init__(self, lambda_dice=1.0, lambda_bce=1.0):
        super().__init__()
        self.dice_loss = DiceLoss(softmax=True, to_onehot_y=True)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.lambda_dice = lambda_dice
        self.lambda_bce = lambda_bce

    def forward(self, pred, target):
        # 确保模型输出和标签的空间维度一致
        if pred.shape[-2:] != target.shape[-2:]:
            target_shape = ensure_interpolate_size(target.shape[-2:])
            pred = F.interpolate(pred, size=target_shape, mode='bilinear', align_corners=False)
        dice_l = self.dice_loss(pred, target)
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        bce_l = self.bce_loss(pred, target_one_hot)
        return self.lambda_dice * dice_l + self.lambda_bce * bce_l

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
    def forward(self, pred, target):
        # pred/target: (B, 1, H, W), 归一化到[0,1]
        ssim_vals = []
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        for i in range(pred.shape[0]):
            ssim_val = skimage_ssim(
                np.squeeze(pred_np[i]), np.squeeze(target_np[i]),
                data_range=1.0, win_size=self.window_size
            )
            ssim_vals.append(ssim_val)
        return 1.0 - torch.tensor(ssim_vals, dtype=pred.dtype, device=pred.device).mean()

class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features[:16].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.resize = resize
        self.norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    def forward(self, pred, target):
        device = pred.device  # 保证VGG和输入在同一设备
        self.vgg = self.vgg.to(device)
        # pred/target: (B, 1, H, W) or (B, H, W)
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)
        pred_rgb = pred.repeat(1, 3, 1, 1)
        target_rgb = target.repeat(1, 3, 1, 1)
        if self.resize:
            pred_rgb = F.interpolate(pred_rgb, size=(224, 224), mode='bilinear', align_corners=False)
            target_rgb = F.interpolate(target_rgb, size=(224, 224), mode='bilinear', align_corners=False)
        pred_rgb = self.norm(pred_rgb)
        target_rgb = self.norm(target_rgb)
        feat_pred = self.vgg(pred_rgb)
        feat_target = self.vgg(target_rgb)
        return F.l1_loss(feat_pred, feat_target)

# --- 损失函数注册表 ---
LOSS_FUNCTIONS = {
    "DiceCELoss": DiceCELoss,
    "DiceBCELoss": DiceBCELoss,
    "CrossEntropyLoss": nn.CrossEntropyLoss,
}

def get_loss_function(name: str) -> nn.Module:
    if name not in LOSS_FUNCTIONS:
        raise ValueError(f"不支持的loss function: {name}，可选: {list(LOSS_FUNCTIONS.keys())}")
    return LOSS_FUNCTIONS[name]()

def train_one_epoch(model, dataloader, optimizer, criterion, device, target_size=None):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for batch in progress_bar:
        # 处理不同的批次格式
        if isinstance(batch, (list, tuple)):
            if len(batch) == 2:
                images, targets = batch
            elif len(batch) == 3:
                images, targets, dataset_names = batch
            else:
                raise ValueError(f"意外的批次格式，期望2或3个元素，得到{len(batch)}个")
        else:
            # 如果batch是字典格式
            if 'image' in batch and 'mask' in batch:
                # 分割任务格式
                images = batch['image']
                targets = batch['mask']
            elif 'input' in batch and 'target' in batch:
                # 图像清晰度任务格式
                images = batch['input']
                targets = batch['target']
            else:
                raise ValueError(f"无法识别的批次格式: {list(batch.keys())}")
        
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        
        # 处理不同的输出格式
        if isinstance(outputs, dict):
            if 'map' in outputs:
                # 图像清晰度任务
                pred = outputs['map']
            else:
                # 其他任务，使用第一个值
                pred = list(outputs.values())[0]
        else:
            pred = outputs
        
        # 确保预测和目标尺寸一致
        if target_size is not None and pred.shape[2:] != target_size:
            pred = F.interpolate(pred, size=target_size, mode='bilinear', align_corners=False)
        elif pred.shape[2:] != targets.shape[2:]:
            # 如果没有指定target_size，则调整到目标尺寸
            target_shape = ensure_interpolate_size(targets.shape[2:])
            pred = F.interpolate(pred, size=target_shape, mode='bilinear', align_corners=False)
            
        loss = criterion(pred, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def dice_score(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    计算每个类别的Dice系数
    Args:
        preds: 预测结果 (B, H, W)
        targets: 真实标签 (B, H, W)
        num_classes: 类别数量
    Returns:
        每个类别的Dice系数 (num_classes,)
    """
    if preds.shape != targets.shape:
        # 调整预测结果的尺寸以匹配目标
        target_shape = ensure_interpolate_size(targets.shape[1:])
        preds = torch.nn.functional.interpolate(
            preds.unsqueeze(1).float(),
            size=target_shape,
            mode='nearest'
        ).squeeze(1).long()
    
    dice_scores = []
    for i in range(num_classes):
        pred_class = (preds == i)
        target_class = (targets == i)
        intersection = (pred_class & target_class).sum().float()
        union = pred_class.sum() + target_class.sum()
        if union == 0:
            dice_scores.append(torch.tensor(1.0, device=preds.device))  # 如果该类别不存在，则dice=1
        else:
            dice = (2.0 * intersection) / (union + 1e-6)  # 添加小量防止除零
            dice_scores.append(dice)
    return torch.stack(dice_scores)

def evaluate_model(model, dataloader, device, num_classes):
    """
    在验证集上评估模型
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        num_classes: 类别数量
    Returns:
        评估结果字典
    """
    model.eval()
    all_dice_scores = []  # 存储所有样本的每个类别dice分数
    dataset_dice_scores = {}  # 按数据集分组的dice scores
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # 处理不同的批次格式
            if isinstance(batch, (list, tuple)):
                if len(batch) == 2:
                    images, targets = batch
                    dataset_names = None
                elif len(batch) == 3:
                    images, targets, dataset_names = batch
                else:
                    raise ValueError(f"意外的批次格式，期望2或3个元素，得到{len(batch)}个")
            else:
                # 如果batch是字典格式
                if 'image' in batch and 'mask' in batch:
                    # 分割任务格式
                    images = batch['image']
                    targets = batch['mask']
                    dataset_names = None
                elif 'input' in batch and 'target' in batch:
                    # 图像清晰度任务格式
                    images = batch['input']
                    targets = batch['target']
                    dataset_names = None
                else:
                    raise ValueError(f"无法识别的批次格式: {list(batch.keys())}")
            
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            
            # 处理不同的输出格式
            if isinstance(outputs, dict):
                if 'map' in outputs:
                    # 图像清晰度任务
                    preds = outputs['map']
                else:
                    # 其他任务，使用第一个值
                    preds = list(outputs.values())[0]
            else:
                preds = outputs
            
            # 对于图像清晰度任务，直接计算损失
            if preds.shape[1] == 1 and targets.shape[1] == 1:
                # 图像清晰度任务，计算MSE损失
                mse_loss = F.mse_loss(preds, targets)
                # 对于图像清晰度任务，我们创建一个假的dice分数数组
                fake_dice = torch.tensor([mse_loss.item()] * num_classes, device=device)
                all_dice_scores.append(fake_dice)
            else:
                # 分割任务，计算Dice分数
                preds = torch.argmax(preds, dim=1)
                dice_scores = dice_score(preds, targets, num_classes)
                all_dice_scores.append(dice_scores)
            
            # 按数据集分组（如果有的话）
            if dataset_names is not None:
                for i, dataset_name in enumerate(dataset_names):
                    if dataset_name not in dataset_dice_scores:
                        dataset_dice_scores[dataset_name] = []
                    if preds.shape[1] == 1 and targets.shape[1] == 1:
                        # 图像清晰度任务
                        mse_loss = F.mse_loss(preds[i:i+1], targets[i:i+1])
                        fake_dice = torch.tensor([mse_loss.item()] * num_classes, device=device)
                        dataset_dice_scores[dataset_name].append(fake_dice)
                    else:
                        # 分割任务
                        dice_scores = dice_score(preds[i:i+1], targets[i:i+1], num_classes)
                        dataset_dice_scores[dataset_name].append(dice_scores)
    
    # 计算每个类别的平均dice分数
    if all_dice_scores:
        all_dice_tensor = torch.stack(all_dice_scores)  # (num_samples, num_classes)
        mean_dice_scores = all_dice_tensor.mean(dim=0).cpu().numpy()  # (num_classes,)
    else:
        mean_dice_scores = np.zeros(num_classes)
    
    # 计算每个数据集的每个类别平均分数
    dataset_avg_scores = {}
    for dataset_name, scores in dataset_dice_scores.items():
        if scores:
            dataset_tensor = torch.stack(scores)  # (num_samples, num_classes)
            dataset_avg_scores[dataset_name] = dataset_tensor.mean(dim=0).cpu().numpy()  # (num_classes,)
        else:
            dataset_avg_scores[dataset_name] = np.zeros(num_classes)
    
    # 从base配置导入类别名称
    from configs import base as cfg_base
    class_names = [cfg_base.CLASS_NAMES[i] for i in range(num_classes)]
    
    return {
        'mean_dice_scores': mean_dice_scores,
        'class_names': class_names,
        'dataset_scores': dataset_avg_scores,
        'overall_score': mean_dice_scores.mean(),
        'total_samples': len(all_dice_scores)
    }

def pretty_print_metrics(results: dict):
    print("\n--- Evaluation Results ---")
    # 打印整体指标
    mean_dice_scores = results['mean_dice_scores']
    class_names = results['class_names']
    print(f"\n[Overall Metrics]")
    for class_name, dice_score in zip(class_names, mean_dice_scores):
        print(f"  - {class_name}: {dice_score:.4f}")
    overall_mean_dice = mean_dice_scores.mean()
    print(f"  - 平均Dice: {overall_mean_dice:.4f}")
    # 打印按数据集分组的指标
    if results['dataset_scores']:
        print(f"\n[Per-Dataset Metrics]")
        for dataset_name, dataset_scores in results['dataset_scores'].items():
            print(f"  - Dataset: {dataset_name}")
            for class_name, dice_score in zip(class_names, dataset_scores):
                print(f"    - {class_name}: {dice_score:.4f}")
            dataset_mean = dataset_scores.mean()
            print(f"    - 平均Dice: {dataset_mean:.4f}")
    print("\n--------------------------\n") 

def build_model(encoder_name: str = "unet", decoder_name: str = "unet", **kwargs):
    """
    构建模型
    
    Args:
        encoder_name: 编码器名称
        decoder_name: 解码器名称
        **kwargs: 其他参数
    
    Returns:
        构建的模型
    """
    # 对于图像清晰度任务，使用专门的模型构建函数
    from models import build_image_sharpness_model
    return build_image_sharpness_model(encoder_name, decoder_name, **kwargs)

def build_optimizer(model_parameters, optimizer_config: dict, lr: float):
    """
    构建优化器
    
    Args:
        model_parameters: 模型参数
        optimizer_config: 优化器配置
        lr: 学习率
    
    Returns:
        优化器
    """
    optimizer_type = optimizer_config.get("type", "AdamW")
    weight_decay = optimizer_config.get("weight_decay", 1e-4)
    
    if optimizer_type.lower() == "adamw":
        return torch.optim.AdamW(model_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == "adam":
        return torch.optim.Adam(model_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == "sgd":
        momentum = optimizer_config.get("momentum", 0.9)
        return torch.optim.SGD(model_parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_type}")

def build_loss(loss_config: dict):
    """
    构建损失函数
    
    Args:
        loss_config: 损失函数配置
    
    Returns:
        损失函数
    """
    loss_type = loss_config.get("type", "L1")
    
    if loss_type.lower() == "l1":
        return nn.L1Loss()
    elif loss_type.lower() == "mse":
        return nn.MSELoss()
    elif loss_type.lower() == "l1_mse":
        class L1MSELoss(nn.Module):
            def __init__(self, lambda_l1=1.0, lambda_mse=1.0):
                super().__init__()
                self.l1_loss = nn.L1Loss()
                self.mse_loss = nn.MSELoss()
                self.lambda_l1 = lambda_l1
                self.lambda_mse = lambda_mse
                
            def forward(self, pred, target):
                l1 = self.l1_loss(pred, target)
                mse = self.mse_loss(pred, target)
                return self.lambda_l1 * l1 + self.lambda_mse * mse
        
        lambda_l1 = loss_config.get("lambda_l1", 1.0)
        lambda_mse = loss_config.get("lambda_mse", 1.0)
        return L1MSELoss(lambda_l1=lambda_l1, lambda_mse=lambda_mse)
    else:
        raise ValueError(f"不支持的损失函数类型: {loss_type}")

class WarmupCosineLRScheduler:
    """学习率调度器"""
    
    def __init__(self, optimizer, warmup_epochs: int = 5, max_epochs: int = 200, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.current_epoch = 0
        
    def step(self):
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            # 预热阶段：线性增长
            lr_scale = self.current_epoch / self.warmup_epochs
        else:
            # 余弦退火阶段
            progress = (self.current_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lr_scale = 0.5 * (1 + np.cos(np.pi * progress))
        
        # 更新学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = max(self.min_lr, param_group['lr'] * lr_scale)


class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=20, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        
        return self.counter >= self.patience 