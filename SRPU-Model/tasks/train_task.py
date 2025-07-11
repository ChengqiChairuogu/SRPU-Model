# self_segmentation/tasks/train_task.py
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
    from configs.train import train_config as cfg_train # <--- 导入新的训练配置
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
        from configs import base as cfg_base; from configs import json_config as cfg_json_gen; from configs.train import train_config as cfg_train; from datasets.sem_segmentation_dataset import SemSegmentationDataset; from utils.augmentations_utils import build_augmentations; from models.encoders.unet_encoder import UNetEncoder; from models.decoders.unet_decoder import UNetDecoder; from models.segmentation_unet import SegmentationUNet; from utils.training_utils import DiceBCELoss, train_one_epoch, validate_one_epoch
    else: raise

def main_train():
    project_root = cfg_base.PROJECT_ROOT.resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Training Task: {cfg_train.TASK_NAME} ---")
    print(f"Using device: {device}")

    # --- 数据加载 ---
    train_augs = build_augmentations(is_train=True)
    val_augs = build_augmentations(is_train=False)
    train_dataset = SemSegmentationDataset(json_file_identifier=cfg_train.TRAIN_JSON_NAME, project_root=project_root, input_depth_from_config=cfg_base.INPUT_DEPTH, class_mapping_from_config=cfg_base.RGB_TO_CLASS, json_dir_name_relative_to_project=cfg_json_gen.JSON_OUTPUT_DIR_NAME, augmentations=train_augs)
    val_dataset = SemSegmentationDataset(json_file_identifier=cfg_train.VAL_JSON_NAME, project_root=project_root, input_depth_from_config=cfg_base.INPUT_DEPTH, class_mapping_from_config=cfg_base.RGB_TO_CLASS, json_dir_name_relative_to_project=cfg_json_gen.JSON_OUTPUT_DIR_NAME, augmentations=val_augs)
    train_loader = DataLoader(train_dataset, batch_size=cfg_train.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=cfg_train.BATCH_SIZE*2, shuffle=False, num_workers=4)

    # --- 模型设置 (随机初始化) ---
    encoder = UNetEncoder(in_channels=cfg_base.INPUT_DEPTH, base_c=64)
    decoder = UNetDecoder(encoder.get_channels(), out_channels=cfg_base.NUM_CLASSES * cfg_base.INPUT_DEPTH, bilinear=cfg_base.BILINEAR)
    model = SegmentationUNet(encoder, decoder).to(device)

    # --- 优化器与损失函数 ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg_train.LEARNING_RATE, weight_decay=cfg_train.WEIGHT_DECAY)
    criterion = DiceBCELoss()

    # --- 日志与检查点 ---
    checkpoint_dir = project_root / "models" / "checkpoints" / cfg_train.TASK_NAME
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_checkpoint_path = checkpoint_dir / "latest_checkpoint.pth"
    best_checkpoint_path = checkpoint_dir / "best_model.pth"
    log_dir = project_root / "logs" / cfg_train.TASK_NAME
    writer = SummaryWriter(log_dir=str(log_dir))

    # --- 训练循环 ---
    best_metric = 0.0
    for epoch in range(cfg_train.NUM_EPOCHS):
        avg_train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        avg_val_loss, avg_val_dice = validate_one_epoch(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{cfg_train.NUM_EPOCHS} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}")
        writer.add_scalar('Loss/Train', avg_train_loss, epoch + 1)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch + 1)
        writer.add_scalar('Metric/Validation_Dice', avg_val_dice, epoch + 1)

        if avg_val_dice > best_metric:
            best_metric = avg_val_dice
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"  -> New best model saved with Dice: {best_metric:.4f}")
    
    writer.close()
    print("Training finished.")

if __name__ == '__main__':
    main_train()
