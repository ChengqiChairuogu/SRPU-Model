# self_segmentation/utils/training_utils.py
import torch
import torch.nn as nn
from tqdm import tqdm

# --- 损失函数 ---
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # inputs: (B, C, H, W) - logits
        # targets: (B, C, H, W) - long or float
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1).float() # Ensure targets are float
        
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = nn.functional.binary_cross_entropy(inputs, targets, reduction='mean')
        return BCE + dice_loss

# --- 评估指标 ---
def dice_coefficient(preds, targets, smooth=1):
    preds = torch.sigmoid(preds) > 0.5
    preds = preds.view(-1).float()
    targets = targets.view(-1).float()
    intersection = (preds * targets).sum()
    return (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)

# --- 训练和验证循环 ---
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    loop = tqdm(dataloader, desc=f"Training")
    for images, masks in loop:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        
        # 注意: 这里的损失计算需要与您的模型输出和目标掩码的形状匹配
        # 假设损失函数可以处理 (B, D*C, H, W) vs (B, D, H, W)
        # 您可能需要根据实际情况调整
        loss = criterion(outputs, masks)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    return total_loss / len(dataloader)

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    with torch.no_grad():
        loop = tqdm(dataloader, desc=f"Validation")
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            dice = dice_coefficient(outputs, masks)
            
            total_loss += loss.item()
            total_dice += dice.item()
            loop.set_postfix(loss=loss.item(), dice=total_dice/len(loop))
            
    avg_loss = total_loss / len(dataloader)
    avg_dice = total_dice / len(dataloader)
    return avg_loss, avg_dice
