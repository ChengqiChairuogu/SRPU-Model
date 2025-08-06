import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import os
import copy

# 添加项目根目录到sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from datasets.sem_datasets import SemSegmentationDataset
from models.segmentation_unet import SegmentationUNet
from utils.training_util import get_loss_function, train_one_epoch, evaluate_model, pretty_print_metrics
from utils.logging_util import Logger
from configs.train import multistage_train_config as cfg

def filter_dataset_by_names(dataset, allowed_names):
    indices = [i for i, s in enumerate(dataset.samples) if s['dataset'] in allowed_names]
    dataset.samples = [dataset.samples[i] for i in indices]
    return dataset

def main_multistage_train():
    print("--- 多阶段有监督训练任务开始 ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prev_model_path = None
    for i, stage in enumerate(cfg.STAGES):
        print(f"\n=== 阶段 {i+1}: {stage['name']} ===")
        # 数据集加载
        train_dataset = SemSegmentationDataset(
            json_file_identifier=stage['train_json'],
            project_root=project_root,
            split='train'
        )
        val_dataset = SemSegmentationDataset(
            json_file_identifier=stage['val_json'],
            project_root=project_root,
            split='val',
            return_dataset_name=True
        )
        train_dataset = filter_dataset_by_names(train_dataset, stage['datasets'])
        val_dataset = filter_dataset_by_names(val_dataset, stage['datasets'])
        train_loader = DataLoader(train_dataset, batch_size=stage['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=stage['batch_size'], shuffle=False, num_workers=2)
        # 权重加载
        if i == 0:
            # 第一阶段：检查是否有预训练模型配置
            if hasattr(cfg, 'USE_PRETRAINED_MODEL') and cfg.USE_PRETRAINED_MODEL:
                pretrained_model_path = stage.get('pretrained_model_path', cfg.PRETRAINED_MODEL_PATH)
                if pretrained_model_path and Path(pretrained_model_path).exists():
                    print(f"加载预训练模型: {pretrained_model_path}")
                    model = SegmentationUNet(
                        encoder_name=cfg.ENCODER_NAME,
                        decoder_name=cfg.DECODER_NAME,
                        n_channels=3,
                        n_classes=3
                    ).to(device)
                    checkpoint = torch.load(pretrained_model_path, map_location=device)
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                        print("预训练模型权重加载成功")
                    elif 'state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['state_dict'])
                        print("预训练模型权重加载成功")
                    else:
                        model.load_state_dict(checkpoint)
                        print("预训练模型权重加载成功")
                else:
                    print("未找到预训练模型，使用随机初始化")
                    model = SegmentationUNet(
                        encoder_name=cfg.ENCODER_NAME,
                        decoder_name=cfg.DECODER_NAME,
                        n_channels=3,
                        n_classes=3
                    ).to(device)
            else:
                # 使用随机初始化
                model = SegmentationUNet(
                    encoder_name=cfg.ENCODER_NAME,
                    decoder_name=cfg.DECODER_NAME,
                    n_channels=3,
                    n_classes=3
                ).to(device)
        else:
            # 后续阶段：使用前一阶段的模型权重
            if prev_model_path and prev_model_path.exists():
                print(f"加载前一阶段模型权重: {prev_model_path}")
                model = SegmentationUNet(
                    encoder_name=cfg.ENCODER_NAME,
                    decoder_name=cfg.DECODER_NAME,
                    n_channels=3,
                    n_classes=3
                ).to(device)
                checkpoint = torch.load(prev_model_path, map_location=device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print("前一阶段模型权重加载成功")
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                    print("前一阶段模型权重加载成功")
                else:
                    model.load_state_dict(checkpoint)
                    print("前一阶段模型权重加载成功")
            else:
                print("未找到前一阶段模型，使用随机初始化")
                model = SegmentationUNet(
                    encoder_name=cfg.ENCODER_NAME,
                    decoder_name=cfg.DECODER_NAME,
                    n_channels=3,
                    n_classes=3
                ).to(device)
        criterion = get_loss_function(cfg.LOSS_FUNCTION)
        optimizer = torch.optim.AdamW(model.parameters(), lr=stage['base_lr'], weight_decay=cfg.WEIGHT_DECAY)
        logger = Logger(cfg.log_config)
        best_val_dice = 0.0
        
        # 创建模型保存目录
        model_save_dir = project_root / cfg.MODEL_SAVE_DIR
        model_save_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置模型保存路径
        best_model_path = model_save_dir / f"{stage['name']}_best_model.pth"
        last_model_path = model_save_dir / f"{stage['name']}_last_model.pth"
        checkpoint_path = model_save_dir / f"{stage['name']}_checkpoint.pth"
        for epoch in range(1, stage['num_epochs'] + 1):
            train_loss = train_one_epoch(model, dataloader=train_loader, optimizer=optimizer, criterion=criterion, device=device)
            evaluation_results = evaluate_model(model, dataloader=val_loader, device=device, num_classes=3)
            mean_dice_scores = evaluation_results['mean_dice_scores']
            overall_mean_dice = mean_dice_scores.mean()
            log_dict = {"train_loss": train_loss, "val_dice": overall_mean_dice}
            for class_name, dice in zip(evaluation_results['class_names'], mean_dice_scores):
                log_dict[f"val/all/{class_name}_dice"] = dice
            for dataset_name, dataset_scores in evaluation_results.get('dataset_scores', {}).items():
                for class_name, dice in zip(evaluation_results['class_names'], dataset_scores):
                    log_dict[f"val/{dataset_name}/{class_name}_dice"] = dice
            logger.log(log_dict, step=epoch)
            print(f"Epoch {epoch} - Val Dice: {overall_mean_dice:.4f}")
            # 输出详细dice
            if epoch == 1 or epoch % cfg.DICE_EVAL_EPOCH_INTERVAL == 0:
                pretty_print_metrics(evaluation_results)
            # 保存最佳模型
            if cfg.SAVE_BEST_MODEL and overall_mean_dice > best_val_dice:
                best_val_dice = overall_mean_dice
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_dice': best_val_dice,
                    'train_loss': train_loss,
                    'config': {
                        'encoder_name': cfg.ENCODER_NAME,
                        'decoder_name': cfg.DECODER_NAME,
                        'n_channels': 3,
                        'n_classes': 3
                    }
                }, best_model_path)
                print(f"新的最佳模型已保存到: {best_model_path}")
                print(f"Val Dice: {best_val_dice:.4f}")
            
            # 保存最后一个epoch的模型
            if cfg.SAVE_LAST_MODEL and epoch == stage['num_epochs']:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_dice': overall_mean_dice,
                    'train_loss': train_loss,
                    'config': {
                        'encoder_name': cfg.ENCODER_NAME,
                        'decoder_name': cfg.DECODER_NAME,
                        'n_channels': 3,
                        'n_classes': 3
                    }
                }, last_model_path)
                print(f"最后一个epoch模型已保存到: {last_model_path}")
            
            # 保存训练检查点
            if cfg.SAVE_CHECKPOINT:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_dice': overall_mean_dice,
                    'train_loss': train_loss,
                    'config': {
                        'encoder_name': cfg.ENCODER_NAME,
                        'decoder_name': cfg.DECODER_NAME,
                        'n_channels': 3,
                        'n_classes': 3
                    }
                }, checkpoint_path)
        logger.close()
        prev_model_path = best_model_path
        print(f"阶段 {stage['name']} 完成，最佳模型保存在: {best_model_path}")
    print("--- 多阶段有监督训练任务全部完成 ---")

if __name__ == '__main__':
    main_multistage_train() 