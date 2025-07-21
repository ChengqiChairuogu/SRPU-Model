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
from utils.training_utils import get_loss_function, train_one_epoch, evaluate_model, pretty_print_metrics
from utils.logger import Logger
from configs.finetune import multistage_finetune_config as cfg

def filter_dataset_by_names(dataset, allowed_names):
    # 只保留属于allowed_names的数据
    indices = [i for i, s in enumerate(dataset.samples) if s['dataset'] in allowed_names]
    dataset.samples = [dataset.samples[i] for i in indices]
    return dataset

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

def create_model_with_pretrained_encoder(encoder_name, decoder_name, pretrained_encoder_path=None):
    model = SegmentationUNet(
        encoder_name=encoder_name,
        decoder_name=decoder_name,
        n_channels=3,
        n_classes=3
    )
    if pretrained_encoder_path is not None and Path(pretrained_encoder_path).exists():
        print(f"加载预训练编码器权重: {pretrained_encoder_path}")
        pretrained_state_dict = torch.load(pretrained_encoder_path, map_location='cpu', weights_only=False)
        current_state_dict = model.state_dict()
        for key in pretrained_state_dict.keys():
            if key in current_state_dict:
                current_state_dict[key] = pretrained_state_dict[key]
        model.load_state_dict(current_state_dict)
        print("预训练编码器权重加载成功")
    else:
        print("未找到预训练编码器权重，使用随机初始化")
    return model

def main_multistage_finetune():
    print("--- 多阶段微调任务开始 ---")
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
        # 数据集筛选
        train_dataset = filter_dataset_by_names(train_dataset, stage['datasets'])
        val_dataset = filter_dataset_by_names(val_dataset, stage['datasets'])
        train_loader = DataLoader(train_dataset, batch_size=stage['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=stage['batch_size'], shuffle=False, num_workers=2)
        # 权重加载
        if i == 0:
            pretrained_encoder_path = stage.get('pretrained_encoder_path', None)
        else:
            pretrained_encoder_path = prev_model_path
        model = create_model_with_pretrained_encoder(
            encoder_name=cfg.ENCODER_NAME,
            decoder_name=cfg.DECODER_NAME,
            pretrained_encoder_path=pretrained_encoder_path
        ).to(device)
        # 损失与优化器
        criterion = get_loss_function(cfg.LOSS_FUNCTION)
        if stage['finetune_mode'] == 'finetune_frozen':
            for param in model.encoder.parameters():
                param.requires_grad = False
            optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=stage['base_lr'], weight_decay=cfg.WEIGHT_DECAY)
        elif stage['finetune_mode'] == 'finetune_differential':
            optimizer = torch.optim.AdamW([
                {"params": model.encoder.parameters(), "lr": stage.get('encoder_lr', stage['base_lr'])},
                {"params": model.decoder.parameters(), "lr": stage['base_lr']}
            ], weight_decay=cfg.WEIGHT_DECAY)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=stage['base_lr'], weight_decay=cfg.WEIGHT_DECAY)
        # 日志
        logger = Logger(cfg.log_config)
        best_val_dice = 0.0
        best_model_path = project_root / f"models/checkpoints/multistage_{stage['name']}_best_model.pth"
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
            if overall_mean_dice > best_val_dice:
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
                print(f"新的最佳模型已保存! Val Dice: {best_val_dice:.4f}")
        logger.close()
        prev_model_path = best_model_path
        print(f"阶段 {stage['name']} 完成，最佳模型保存在: {best_model_path}")
    print("--- 多阶段微调任务全部完成 ---")

if __name__ == '__main__':
    main_multistage_finetune() 