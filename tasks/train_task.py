import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys
from pathlib import Path
import json

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from datasets.sem_datasets import SemSegmentationDataset
from models.segmentation_unet import SegmentationUNet
from utils.training_utils import get_loss_function, train_one_epoch, evaluate_model, pretty_print_metrics
from utils.logger import Logger
from configs.train import train_config as cfg_train

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

def main_train():
    """
    监督分割训练任务的主函数。
    """
    print("--- 开始监督分割训练任务 ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # --- 创建检查点和模型保存目录 ---
    project_root_path = Path(__file__).resolve().parent.parent
    checkpoint_dir = project_root_path / cfg_train.CHECKPOINT_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = project_root_path / cfg_train.RESUMABLE_CHECKPOINT_PATH
    best_model_path = project_root_path / cfg_train.BEST_MODEL_CHECKPOINT_PATH

    # ---1
    train_dataset = SemSegmentationDataset(
        json_file_identifier=cfg_train.TRAIN_JSON_NAME,
        project_root=project_root_path,
        split='train'
    )
    train_loader = DataLoader(train_dataset, batch_size=cfg_train.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    # 为验证创建单独的数据集和加载器，并要求返回数据集名称
    val_dataset = SemSegmentationDataset(
        json_file_identifier=cfg_train.VAL_JSON_NAME,
        project_root=project_root_path,
        split='val',
        return_dataset_name=True
    )
    val_loader = DataLoader(val_dataset, batch_size=cfg_train.BATCH_SIZE, shuffle=False, num_workers=2)

    # --- 2. 创建模型 ---
    print(f"--- 正在创建模型: Encoder: {cfg_train.ENCODER_NAME}, Decoder: {cfg_train.DECODER_NAME} ---")
    model = SegmentationUNet(
        encoder_name=cfg_train.ENCODER_NAME,
        decoder_name=cfg_train.DECODER_NAME,
        n_channels=3, 
        n_classes=3   
    ).to(device)

    # --- 3
    criterion = get_loss_function(cfg_train.LOSS_FUNCTION)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg_train.LEARNING_RATE, weight_decay=cfg_train.WEIGHT_DECAY)

    # --- 4. 断点续训：尝试加载检查点 ---
    start_epoch = 1
    best_val_dice = 0.0
    best_train_loss = float('inf')
    if cfg_train.RESUME_FROM_CHECKPOINT and os.path.exists(checkpoint_path):
        start_epoch, best_train_loss, best_val_dice = load_checkpoint(model, optimizer, checkpoint_path, device)
        start_epoch += 1  # 从下一个epoch开始

    # --- 5. 训练循环 ---
    epochs = cfg_train.NUM_EPOCHS
    print(f"将从 Epoch {start_epoch} 开始训练，共 {epochs} 个 Epochs...")

    logger = Logger(cfg_train.log_config)

    for epoch in range(start_epoch, epochs + 1):
        train_loss = train_one_epoch(model, dataloader=train_loader, optimizer=optimizer, criterion=criterion, device=device)
        print(f"\nEpoch {epoch} 完成. 平均训练损失: {train_loss:.4f}")

        # --- 每个epoch都评估 ---
        print(f"--- Running evaluation for Epoch {epoch} ---")
        evaluation_results = evaluate_model(model, dataloader=val_loader, device=device, num_classes=3)
        
        mean_dice_scores = evaluation_results['mean_dice_scores']
        overall_mean_dice = mean_dice_scores.mean()
        # 日志记录
        log_dict = {"train_loss": train_loss, "val_dice": overall_mean_dice}
        # 记录每个class整体dice
        for class_name, dice in zip(evaluation_results['class_names'], mean_dice_scores):
            log_dict[f"val/all/{class_name}_dice"] = dice
        # 记录每个dataset的各类别dice
        for dataset_name, dataset_scores in evaluation_results.get('dataset_scores', {}).items():
            for class_name, dice in zip(evaluation_results['class_names'], dataset_scores):
                log_dict[f"val/{dataset_name}/{class_name}_dice"] = dice
        logger.log(log_dict, step=epoch)
        print(f"Epoch {epoch} - 总Val Dice: {overall_mean_dice:.4f}")
        
        # --- 保存检查点 ---
        save_checkpoint(model, optimizer, epoch, train_loss, overall_mean_dice, checkpoint_path)
        
        # --- 保存最佳模型 ---
        if overall_mean_dice > best_val_dice:
            best_val_dice = overall_mean_dice
            torch.save({
              'epoch': epoch,
               'model_state_dict': model.state_dict(),
              'val_dice': best_val_dice,
           'train_loss': train_loss,
          'config': {
                'encoder_name': cfg_train.ENCODER_NAME,
                 'decoder_name': cfg_train.DECODER_NAME,
                   'n_channels': 3,
                   'n_classes': 3
                }
            }, best_model_path)
            print(f"新的最佳模型已保存! Val Dice: {best_val_dice:.4f}")
        
        # 每隔指定epoch数才输出详细的dice
        if epoch == 1 or epoch % cfg_train.DICE_EVAL_EPOCH_INTERVAL == 0:
            pretty_print_metrics(evaluation_results)

    print("--- 训练任务完成 ---")
    print(f"最佳验证Dice: {best_val_dice:.4f}")
    print(f"最佳模型保存在: {best_model_path}")
    logger.close()

if __name__ == '__main__':
    main_train()