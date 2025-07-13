import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ==============================================================================
#                      修正后的损失函数
# ==============================================================================
class DiceBCELoss(nn.Module):
    """
    一个更健壮的Dice-BCE损失函数，用于多分类语义分割。
    它现在可以自动处理模型输出和目标尺寸不匹配的问题。
    """
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, logits, targets, smooth=1e-6):
        # **关键修正**: 检查并统一空间尺寸
        # 如果模型输出(logits)和真值掩码(targets)的尺寸不匹配
        if logits.shape[-2:] != targets.shape[-2:]:
            # 使用双线性插值将logits的尺寸调整为和targets一致
            logits = F.interpolate(logits, size=targets.shape[-2:], mode='bilinear', align_corners=False)

        # 1. 计算 BCE Loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='mean')

        # 2. 计算 Dice Loss
        probs = F.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        dice_score = (2. * intersection + smooth) / (probs_flat.sum() + targets_flat.sum() + smooth)
        dice_loss = 1 - dice_score
        
        # 3. 组合损失
        combined_loss = bce_loss + dice_loss
        
        return combined_loss

# ==============================================================================
#                      训练和验证工具函数
# ==============================================================================

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """训练模型一个 epoch。"""
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for images, masks in progress_bar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        
        outputs = model(images)
        # 现在criterion内部会自动处理尺寸不匹配的问题
        loss = criterion(outputs, masks)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)


def validate_one_epoch(model, dataloader, criterion, device):
    """在验证集上评估模型一个 epoch。"""
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    progress_bar = tqdm(dataloader, desc="Validating", leave=False)

    with torch.no_grad():
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            # criterion内部会自动处理尺寸不匹配
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            # 计算 Dice 分数用于评估
            probs = F.sigmoid(outputs)
            
            # **关键修正**: 在计算评估指标时，也要确保尺寸一致
            if probs.shape[-2:] != masks.shape[-2:]:
                probs = F.interpolate(probs, size=masks.shape[-2:], mode='bilinear', align_corners=False)

            intersection = (probs.view(-1) * masks.view(-1)).sum()
            dice_score = (2. * intersection) / (probs.view(-1).sum() + masks.view(-1).sum() + 1e-6)
            total_dice += dice_score.item()
            
            progress_bar.set_postfix(loss=loss.item(), dice=dice_score.item())

    avg_loss = total_loss / len(dataloader)
    avg_dice = total_dice / len(dataloader)
    
    return avg_loss, avg_dice