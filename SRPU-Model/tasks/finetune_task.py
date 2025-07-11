# self_segmentation/tasks/finetune_task.py
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import sys
import time

# --- 导入 ---
try:
    from configs import base as cfg_base
    from configs import json_config as cfg_json_gen
    from configs.finetune import finetune_config as cfg_finetune # <--- 导入新的微调配置
    from datasets.sem_segmentation_dataset import SemSegmentationDataset
    from utils.augmentations_utils import build_augmentations
    from models.encoders.unet_encoder import UNetEncoder
    from models.decoders.unet_decoder import UNetDecoder
    from models.segmentation_unet import SegmentationUNet
    from utils.training_utils import DiceBCELoss, train_one_epoch, validate_one_epoch
except ImportError:
    # ... (健壮的导入逻辑) ...
    if __name__ == '__main__' and __package__ is None:
        current_script_path = Path(__file__).resolve()
        project_root = current_script_path.parent.parent
        sys.path.insert(0, str(project_root))
        from configs import base as cfg_base; from configs import json_config as cfg_json_gen; from configs.finetune import finetune_config as cfg_finetune; from datasets.sem_segmentation_dataset import SemSegmentationDataset; from utils.augmentations_utils import build_augmentations; from models.encoders.unet_encoder import UNetEncoder; from models.decoders.unet_decoder import UNetDecoder; from models.segmentation_unet import SegmentationUNet; from utils.training_utils import DiceBCELoss, train_one_epoch, validate_one_epoch
    else: raise

def main_finetune():
    project_root = cfg_base.PROJECT_ROOT.resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Finetune Task: {cfg_finetune.TASK_NAME} ---")
    print(f"Using device: {device}")
    print(f"Finetune Mode: {cfg_finetune.FINETUNE_MODE}")

    # --- 数据加载 (与train_task相同) ---
    train_augs = build_augmentations(is_train=True)
    val_augs = build_augmentations(is_train=False)
    train_dataset = SemSegmentationDataset(json_file_identifier=cfg_finetune.TRAIN_JSON_NAME, project_root=project_root, input_depth_from_config=cfg_base.INPUT_DEPTH, class_mapping_from_config=cfg_base.RGB_TO_CLASS, json_dir_name_relative_to_project=cfg_json_gen.JSON_OUTPUT_DIR_NAME, augmentations=train_augs)
    val_dataset = SemSegmentationDataset(json_file_identifier=cfg_finetune.VAL_JSON_NAME, project_root=project_root, input_depth_from_config=cfg_base.INPUT_DEPTH, class_mapping_from_config=cfg_base.RGB_TO_CLASS, json_dir_name_relative_to_project=cfg_json_gen.JSON_OUTPUT_DIR_NAME, augmentations=val_augs)
    train_loader = DataLoader(train_dataset, batch_size=cfg_finetune.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=cfg_finetune.BATCH_SIZE*2, shuffle=False, num_workers=4)

    # --- 模型设置 ---
    encoder = UNetEncoder(in_channels=cfg_base.INPUT_DEPTH, base_c=64)
    decoder = UNetDecoder(encoder.get_channels(), out_channels=cfg_base.NUM_CLASSES * cfg_base.INPUT_DEPTH, bilinear=cfg_base.BILINEAR)
    model = SegmentationUNet(encoder, decoder).to(device)

    # --- 加载预训练权重 ---
    pretrained_path = project_root / cfg_finetune.PRETRAINED_ENCODER_PATH
    if pretrained_path.exists():
        print(f"Loading pretrained weights for encoder from: {pretrained_path}")
        try:
            encoder_state_dict = torch.load(pretrained_path, map_location=device)
            model.encoder.load_state_dict(encoder_state_dict, strict=True)
        except Exception as e:
            print(f"Error loading pretrained weights: {e}. Training from scratch instead.")
    else:
        print(f"Warning: Pretrained encoder not found at {pretrained_path}. Training from scratch instead.")
    
    # --- 设置优化器 (根据微调模式) ---
    if cfg_finetune.FINETUNE_MODE == 'finetune_frozen':
        print("Optimizer setup: Freezing encoder, training decoder only.")
        for param in model.encoder.parameters():
            param.requires_grad = False
        params_to_update = model.decoder.parameters()
        optimizer = torch.optim.AdamW(params_to_update, lr=cfg_finetune.BASE_LEARNING_RATE, weight_decay=cfg_finetune.WEIGHT_DECAY)
    elif cfg_finetune.FINETUNE_MODE == 'finetune_differential':
        print("Optimizer setup: Using differential learning rates.")
        optimizer_params = [
            {'params': model.encoder.parameters(), 'lr': cfg_finetune.ENCODER_LEARNING_RATE},
            {'params': model.decoder.parameters(), 'lr': cfg_finetune.BASE_LEARNING_RATE}
        ]
        optimizer = torch.optim.AdamW(optimizer_params, weight_decay=cfg_finetune.WEIGHT_DECAY)
    else: # 默认情况或配置错误，按常规微调处理
        print("Optimizer setup: Training all parameters with base learning rate.")
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg_finetune.BASE_LEARNING_RATE, weight_decay=cfg_finetune.WEIGHT_DECAY)

    # --- 损失函数和检查点 (与train_task类似) ---
    criterion = DiceBCELoss()
    checkpoint_dir = project_root / "models" / "checkpoints" / cfg_finetune.TASK_NAME
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / "best_model.pth"
    log_dir = project_root / "logs" / cfg_finetune.TASK_NAME
    writer = SummaryWriter(log_dir=str(log_dir))

    # --- 训练与验证循环 (使用工具函数) ---
    best_metric = 0.0
    for epoch in range(cfg_finetune.NUM_EPOCHS):
        avg_train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        avg_val_loss, avg_val_dice = validate_one_epoch(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{cfg_finetune.NUM_EPOCHS} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}")
        writer.add_scalar('Loss/Train', avg_train_loss, epoch + 1)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch + 1)
        writer.add_scalar('Metric/Validation_Dice', avg_val_dice, epoch + 1)

        if avg_val_dice > best_metric:
            best_metric = avg_val_dice
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"  -> New best model saved with Dice: {best_metric:.4f}")

    writer.close()
    print("Finetuning finished.")

if __name__ == '__main__':
    main_finetune()
