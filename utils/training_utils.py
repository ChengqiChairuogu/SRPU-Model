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
            pred = F.interpolate(pred, size=target.shape[-2:], mode='bilinear', align_corners=False)
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
            pred = F.interpolate(pred, size=target.shape[-2:], mode='bilinear', align_corners=False)
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

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for images, masks in progress_bar:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
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
        preds = torch.nn.functional.interpolate(
            preds.unsqueeze(1).float(),
            size=targets.shape[1:],
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
    total_dice_scores = []
    dataset_dice_scores = {}  # 按数据集分组的dice scores
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if len(batch) == 3:  # 如果返回了数据集名称
                images, masks, dataset_names = batch
            else:
                images, masks = batch
                dataset_names = None
            
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            # 计算整体的dice scores
            batch_dice_scores = dice_score(preds, masks, num_classes)
            total_dice_scores.append(batch_dice_scores)
            
            # 如果有数据集名称，按数据集分组计算
            if dataset_names is not None:
                for i, dataset_name in enumerate(dataset_names):
                    if dataset_name not in dataset_dice_scores:
                        dataset_dice_scores[dataset_name] = []
                    single_pred = preds[i].unsqueeze(0)
                    single_mask = masks[i].unsqueeze(0)
                    single_dice_score = dice_score(single_pred, single_mask, num_classes)
                    dataset_dice_scores[dataset_name].append(single_dice_score)
    
    # 计算平均值
    mean_dice_scores = torch.stack(total_dice_scores).mean(dim=0)
    
    results = {
        'mean_dice_scores': mean_dice_scores.cpu().numpy(),
        'class_names': ['carbon', 'SE', 'AM'],  # 与MAPPING注释一致
        'dataset_scores': {}
    }
    
    # 计算每个数据集的平均分数
    if dataset_dice_scores:
        for dataset_name, scores in dataset_dice_scores.items():
            dataset_mean = torch.stack(scores).mean(dim=0)
            results['dataset_scores'][dataset_name] = dataset_mean.cpu().numpy()
    
    return results

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