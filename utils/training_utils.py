import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ==============================================================================
#                      修正后的多类别损失函数
# ==============================================================================
class DiceCELoss(nn.Module):
    """
    专为多类别语义分割设计的 Dice + CrossEntropy 复合损失函数。
    """
    def __init__(self, weight=None, size_average=True, a=0.5, b=0.5):
        super(DiceCELoss, self).__init__()
        self.a = a  # CrossEntropy Loss 的权重
        self.b = b  # Dice Loss 的权重

    def forward(self, logits, targets, smooth=1e-6):
        # 确保 logits 和 targets 的尺寸匹配
        if logits.shape[-2:] != targets.shape[-2:]:
            logits = F.interpolate(logits, size=targets.shape[-2:], mode='bilinear', align_corners=False)

        # 1. 计算 CrossEntropy Loss
        # CrossEntropyLoss 期望的 logits 形状: (N, C, H, W)
        # 期望的 targets 形状: (N, H, W)，值为类别索引
        ce_loss = F.cross_entropy(logits, targets, reduction='mean')

        # 2. 计算 Dice Loss
        # 首先将 logits 转换为概率分布
        probs = F.softmax(logits, dim=1)
        # 将 targets (N, H, W) 转换为 one-hot 编码 (N, C, H, W)
        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()

        intersection = torch.sum(probs * targets_one_hot, dim=(2, 3))
        cardinality = torch.sum(probs + targets_one_hot, dim=(2, 3))

        dice_score = (2. * intersection + smooth) / (cardinality + smooth)
        dice_loss = 1 - dice_score.mean()

        # 3. 组合损失
        combined_loss = self.a * ce_loss + self.b * dice_loss
        return combined_loss

# ==============================================================================
#                      训练和验证工具函数 (保持不变)
# ==============================================================================

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """训练模型一个 epoch。"""
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for images, masks in progress_bar:
        images = images.to(device)
        masks = masks.to(device) # masks 形状应为 (N, H, W)

        optimizer.zero_grad()
        
        outputs = model(images)
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
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            # 计算 Dice 分数用于评估
            probs = F.softmax(outputs, dim=1)
            pred_labels = torch.argmax(probs, dim=1)
            
            # 计算多类别的平均 Dice Score
            dice_per_class = []
            for i in range(outputs.shape[1]): # 遍历每个类别
                pred_i = (pred_labels == i)
                target_i = (masks == i)
                intersection = (pred_i * target_i).sum()
                union = pred_i.sum() + target_i.sum()
                dice = (2. * intersection) / (union + 1e-6)
                dice_per_class.append(dice.item())
            
            avg_dice_score = np.mean(dice_per_class)
            total_dice += avg_dice_score
            
            progress_bar.set_postfix(loss=loss.item(), dice=avg_dice_score)

    avg_loss = total_loss / len(dataloader)
    avg_dice = total_dice / len(dataloader)
    
    return avg_loss, avg_dice