# self_segmentation/tasks/train_task.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import sys
import time
from tqdm import tqdm # 用于显示进度条

# --- 配置与模块导入 ---
try:
    from configs import base as cfg_base
    from configs import json_config as cfg_json_gen
    from configs.finetune import train_config as cfg_train
    from datasets.sem_segmentation_dataset import SemSegmentationDataset
    from utils.augmentations_utils import build_augmentations
    from models.encoders.unet_encoder import UNetEncoder
    from models.decoders.unet_decoder import UNetDecoder
    from models.segmentation_unet import SegmentationUNet
    # 您需要一个存放损失函数和评估指标的 utils 文件
    # from utils.losses import DiceBCELoss, FocalLoss
    # from utils.metrics import dice_coefficient
except ImportError:
    # ... (健壮的导入逻辑) ...
    if __name__ == '__main__' and __package__ is None:
        current_script_path = Path(__file__).resolve()
        project_root = current_script_path.parent.parent
        sys.path.insert(0, str(project_root))
        try:
            from configs import base as cfg_base
            from configs import json_config as cfg_json_gen
            from configs.finetune import train_config as cfg_train
            from datasets.sem_segmentation_dataset import SemSegmentationDataset
            from utils.augmentations_utils import build_augmentations
            from models.encoders.unet_encoder import UNetEncoder
            from models.decoders.unet_decoder import UNetDecoder
            from models.segmentation_unet import SegmentationUNet
        except ImportError as e_inner:
            print(f"Error importing modules in train_task.py: {e_inner}")
            sys.exit(1)
    else:
        raise

# --- 临时的损失和评估函数，用于演示 ---
# 您应该在 utils/losses.py 和 utils/metrics.py 中实现您自己的版本
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = nn.functional.binary_cross_entropy(inputs, targets, reduction='mean')
        return BCE + dice_loss

def dice_coefficient(preds, targets, smooth=1):
    preds = torch.sigmoid(preds) > 0.5
    preds = preds.view(-1).float()
    targets = targets.view(-1).float()
    intersection = (preds * targets).sum()
    return (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)


# --- 主训练函数 ---
def train_and_finetune():
    project_root = cfg_base.PROJECT_ROOT.resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Current Training Mode: {cfg_train.TRAINING_MODE}")

    # --- 1. 数据加载 ---
    train_augs = build_augmentations(is_train=True)
    val_augs = build_augmentations(is_train=False)

    train_dataset = SemSegmentationDataset(
        json_file_identifier=cfg_train.TRAIN_JSON_NAME,
        project_root=project_root,
        input_depth_from_config=cfg_json_gen.INPUT_DEPTH,
        class_mapping_from_config=cfg_base.RGB_TO_CLASS, # 注意，这里用RGB_TO_CLASS
        json_dir_name_relative_to_project=cfg_json_gen.JSON_OUTPUT_DIR_NAME,
        augmentations=train_augs
    )
    val_dataset = SemSegmentationDataset(
        json_file_identifier=cfg_train.VAL_JSON_NAME,
        project_root=project_root,
        input_depth_from_config=cfg_json_gen.INPUT_DEPTH,
        class_mapping_from_config=cfg_base.RGB_TO_CLASS,
        json_dir_name_relative_to_project=cfg_json_gen.JSON_OUTPUT_DIR_NAME,
        augmentations=val_augs
    )
    train_loader = DataLoader(train_dataset, batch_size=cfg_train.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg_train.BATCH_SIZE*2, shuffle=False, num_workers=4)

    # --- 2. 模型设置 ---
    # 实例化编码器和解码器
    encoder = UNetEncoder(in_channels=cfg_base.INPUT_DEPTH, base_c=64)
    # 解码器输出通道数 = 类别数 * 输入深度 (因为模型为每个深度切片都输出一个多类别图)
    decoder = UNetDecoder(encoder.get_channels(), out_channels=cfg_base.NUM_CLASSES * cfg_base.INPUT_DEPTH, bilinear=cfg_base.BILINEAR)
    model = SegmentationUNet(encoder, decoder).to(device)

    # --- 3. 加载预训练权重 (如果需要) ---
    if cfg_train.TRAINING_MODE != 'from_scratch':
        pretrained_path = project_root / cfg_train.PRETRAINED_ENCODER_PATH
        if pretrained_path.exists():
            print(f"Loading SSL pretrained weights for encoder from: {pretrained_path}")
            try:
                encoder_state_dict = torch.load(pretrained_path, map_location=device)
                model.encoder.load_state_dict(encoder_state_dict, strict=True)
            except Exception as e:
                print(f"Error loading pretrained weights: {e}. Training from scratch instead.")
        else:
            print(f"Warning: Pretrained encoder not found at {pretrained_path}. Training from scratch instead.")
    
    # --- 4. 设置优化器 (根据训练模式) ---
    if cfg_train.TRAINING_MODE == 'finetune_frozen':
        print("Optimizer setup: Freezing encoder, training decoder only.")
        for param in model.encoder.parameters():
            param.requires_grad = False
        # 只将需要训练的参数（解码器）传递给优化器
        params_to_update = model.decoder.parameters()
        optimizer = torch.optim.AdamW(params_to_update, lr=cfg_train.BASE_LEARNING_RATE, weight_decay=cfg_train.WEIGHT_DECAY)
    elif cfg_train.TRAINING_MODE == 'finetune_differential':
        print("Optimizer setup: Using differential learning rates for encoder and decoder.")
        # 为编码器和解码器创建不同的参数组
        optimizer_params = [
            {'params': model.encoder.parameters(), 'lr': cfg_train.ENCODER_LEARNING_RATE},
            {'params': model.decoder.parameters(), 'lr': cfg_train.BASE_LEARNING_RATE}
        ]
        optimizer = torch.optim.AdamW(optimizer_params, weight_decay=cfg_train.WEIGHT_DECAY)
    else: # from_scratch
        print("Optimizer setup: Training all parameters with base learning rate.")
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg_train.BASE_LEARNING_RATE, weight_decay=cfg_train.WEIGHT_DECAY)

    # --- 5. 损失函数和检查点 ---
    criterion = DiceBCELoss() # 您可以根据配置动态选择
    
    checkpoint_dir = project_root / "models" / "checkpoints" / cfg_train.TASK_NAME
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_checkpoint_path = checkpoint_dir / "latest_checkpoint.pth"
    best_checkpoint_path = checkpoint_dir / "best_model.pth"
    
    log_dir = project_root / "logs" / cfg_train.TASK_NAME
    writer = SummaryWriter(log_dir=str(log_dir))

    # --- 6. 恢复训练逻辑 ---
    start_epoch = 0
    best_metric = 0.0 # 对于Dice/IoU，越高越好

    if cfg_train.RESUME_FROM_CHECKPOINT and Path(cfg_train.RESUME_FROM_CHECKPOINT).exists():
        resume_path = project_root / cfg_train.RESUME_FROM_CHECKPOINT
        print(f"Resuming training from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_metric = checkpoint.get('best_metric', 0.0)
        print(f"Resumed from epoch {start_epoch-1}. Best metric so far: {best_metric:.4f}.")


    # --- 7. 训练与验证循环 ---
    print(f"Starting training for {cfg_train.NUM_EPOCHS} epochs...")
    for epoch in range(start_epoch, cfg_train.NUM_EPOCHS):
        # --- 训练 ---
        model.train()
        train_loss = 0.0
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg_train.NUM_EPOCHS} [Training]")
        for images, masks in train_loop:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            
            # 您的模型为每个深度输出一个多类别图，而损失函数通常期望 (B, C, H, W) vs (B, 1, H, W) 或 (B, H, W)
            # 您需要根据损失函数和模型输出的实际形状调整这里
            # 假设损失函数可以处理 (B, D*C, H, W) vs (B, D, H, W)
            # 简化：我们假设模型输出是 (B, C, H, W) 且目标是 (B, 1, H, W)
            # 您需要根据您U-Net解码器最终的输出形状调整这里
            # 假设最终输出需要 reshape
            # B, DC, H, W -> B, D, C, H, W
            # outputs = outputs.view(images.size(0), cfg_base.INPUT_DEPTH, cfg_base.NUM_CLASSES, cfg_base.IMAGE_HEIGHT, cfg_base.IMAGE_WIDTH)
            # 这里的损失计算需要与您原始项目的逻辑对齐
            loss = criterion(outputs, masks.float()) # DiceBCELoss通常需要目标是float

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())
        
        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar('Loss/Train', avg_train_loss, epoch + 1)

        # --- 验证 ---
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{cfg_train.NUM_EPOCHS} [Validation]")
            for images, masks in val_loop:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks.float())
                val_loss += loss.item()
                val_dice += dice_coefficient(outputs, masks).item()
                val_loop.set_postfix(loss=loss.item(), dice=val_dice/(len(val_loop)))
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch + 1)
        writer.add_scalar('Metric/Validation_Dice', avg_val_dice, epoch + 1)
        
        print(f"Epoch {epoch+1} Summary: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}")

        # --- 保存检查点 ---
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_metric': best_metric,
        }
        torch.save(checkpoint_data, latest_checkpoint_path) # 保存最新

        if avg_val_dice > best_metric:
            best_metric = avg_val_dice
            print(f"  -> New best validation Dice: {best_metric:.4f}. Saving best model...")
            torch.save(model.state_dict(), best_checkpoint_path) # 只保存模型权重

    writer.close()
    print(f"Training finished. Best validation Dice: {best_metric:.4f}")
    print(f"Best model saved to: {best_checkpoint_path}")


if __name__ == '__main__':
    train_and_finetune()
