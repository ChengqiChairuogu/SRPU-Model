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
from datasets.stratified_sampler import create_stratified_dataloader, get_dataset_indices_from_sem_dataset
from models.segmentation_unet import SegmentationUNet
from utils.training_utils import get_loss_function, train_one_epoch, evaluate_model, pretty_print_metrics
from utils.logger import Logger
from configs.finetune import finetune_config as cfg_finetune

def save_checkpoint(model, optimizer, epoch, train_loss, val_dice, checkpoint_path):
    """保存训练检查点"""
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
    
    # 尝试加载优化器状态，如果参数组数量不匹配则跳过
    try:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("优化器状态加载成功")
    except ValueError as e:
        print(f"警告: 优化器状态加载失败 ({e})，将使用新的优化器状态")
        print("这通常发生在微调模式改变时（如从冻结模式切换到差异化学习率模式）")
    
    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']
    val_dice = checkpoint['val_dice']
    
    print(f"从检查点恢复训练 - Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Dice: {val_dice:.4f}")
    return epoch, train_loss, val_dice

def create_model_with_pretrained_encoder(encoder_name: str, decoder_name: str, pretrained_encoder_path=None):
    """创建模型并加载预训练编码器权重"""
    print(f"--- 正在创建模型: Encoder: {encoder_name}, Decoder: {decoder_name} ---")
    
    # 创建模型
    model = SegmentationUNet(
        encoder_name=encoder_name,
        decoder_name=decoder_name,
        n_channels=3, 
        n_classes=3   
    )
    
    # 加载预训练编码器权重
    if pretrained_encoder_path is not None and pretrained_encoder_path.exists():
        print(f"加载自监督预训练编码器权重: {pretrained_encoder_path}")
        # 加载预训练权重到编码器部分
        pretrained_state_dict = torch.load(pretrained_encoder_path, map_location='cpu', weights_only=False)
        
        # 获取当前模型的编码器部分
        current_state_dict = model.state_dict()
        
        # 只更新编码器部分的权重
        for key in pretrained_state_dict.keys():
            if key in current_state_dict:
                current_state_dict[key] = pretrained_state_dict[key]
        
        model.load_state_dict(current_state_dict)
        print("预训练编码器权重加载成功")
    else:
        print("未找到预训练编码器权重，使用随机初始化")
    
    return model

def main_finetune():
    """
    微调任务的主函数。
    """
    print("--- 开始微调任务 ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # --- 创建检查点和模型保存目录 ---
    project_root_path = Path(__file__).resolve().parent.parent
    checkpoint_dir = project_root_path / cfg_finetune.CHECKPOINT_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = project_root_path / cfg_finetune.RESUMABLE_CHECKPOINT_PATH
    best_model_path = project_root_path / cfg_finetune.BEST_MODEL_CHECKPOINT_PATH

    # --- 1. 数据集和数据加载器 ---
    train_dataset = SemSegmentationDataset(
        json_file_identifier=cfg_finetune.TRAIN_JSON_NAME,
        project_root=project_root_path,
        split='train'
    )
    
    # 暂时使用标准采样器，避免分层采样器的问题
    print("--- 使用标准采样器（临时） ---")
    train_loader = DataLoader(train_dataset, batch_size=cfg_finetune.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    
    # 显示数据集分布信息
    train_dataset_indices = get_dataset_indices_from_sem_dataset(train_dataset)
    print(f"训练数据集分布: {train_dataset_indices}")
    for dataset_name, indices in train_dataset_indices.items():
        print(f"  {dataset_name}: {len(indices)} 个样本")

    # 为验证创建单独的数据集和加载器，并要求返回数据集名称
    val_dataset = SemSegmentationDataset(
        json_file_identifier=cfg_finetune.VAL_JSON_NAME,
        project_root=project_root_path,
        split='val',
        return_dataset_name=True
    )
    val_loader = DataLoader(val_dataset, batch_size=cfg_finetune.BATCH_SIZE, shuffle=False, num_workers=2)

    # --- 2. 创建模型并加载预训练权重 ---
    pretrained_encoder_path = project_root_path / cfg_finetune.PRETRAINED_ENCODER_PATH
    model = create_model_with_pretrained_encoder(
        encoder_name=cfg_finetune.ENCODER_NAME,
        decoder_name=cfg_finetune.DECODER_NAME,
        pretrained_encoder_path=pretrained_encoder_path
    ).to(device)

    # --- 3. 定义损失函数和优化器 ---
    criterion = get_loss_function(cfg_finetune.LOSS_FUNCTION)
    
    # 差异化学习率设置
    if cfg_finetune.FINETUNE_MODE == 'finetune_frozen':
        # 冻结编码器，只训练解码器
        for param in model.encoder.parameters():
            param.requires_grad = False
        optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=cfg_finetune.BASE_LEARNING_RATE, weight_decay=cfg_finetune.WEIGHT_DECAY)
        print("微调模式: 编码器冻结，只训练解码器")
    elif cfg_finetune.FINETUNE_MODE == 'finetune_differential':
        # 编码器和解码器使用不同学习率
        optimizer = torch.optim.AdamW([
            {"params": model.encoder.parameters(), "lr": cfg_finetune.ENCODER_LEARNING_RATE},
            {"params": model.decoder.parameters(), "lr": cfg_finetune.BASE_LEARNING_RATE}
        ], weight_decay=cfg_finetune.WEIGHT_DECAY)
        print(f"微调模式: 差异化学习率 - 编码器: {cfg_finetune.ENCODER_LEARNING_RATE}, 解码器: {cfg_finetune.BASE_LEARNING_RATE}")
    else:
        # 统一学习率
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg_finetune.BASE_LEARNING_RATE, weight_decay=cfg_finetune.WEIGHT_DECAY)
        print(f"微调模式: 统一学习率 - {cfg_finetune.BASE_LEARNING_RATE}")

    # --- 4. 断点续训：尝试加载检查点 ---
    start_epoch = 1
    best_val_dice = 0.0
    best_train_loss = float('inf')
    
    if cfg_finetune.RESUME_FROM_CHECKPOINT and os.path.exists(checkpoint_path):
        start_epoch, best_train_loss, best_val_dice = load_checkpoint(model, optimizer, checkpoint_path, device)
        start_epoch += 1  # 从下一个epoch开始

    # --- 5. 训练循环 ---
    epochs = cfg_finetune.NUM_EPOCHS
    print(f"将从 Epoch {start_epoch} 开始训练，共 {epochs} 个 Epochs...")

    logger = Logger(cfg_finetune.log_config)

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
                    'encoder_name': cfg_finetune.ENCODER_NAME,
                    'decoder_name': cfg_finetune.DECODER_NAME,
                    'n_channels': 3,
                    'n_classes': 3,
                    'finetune_mode': cfg_finetune.FINETUNE_MODE,
                    'pretrained_encoder_path': str(pretrained_encoder_path)
                }
            }, best_model_path)
            print(f"新的最佳模型已保存! Val Dice: {best_val_dice:.4f}")
        
        # 每隔指定epoch数才输出详细的dice
        if epoch == 1 or epoch % 10 == 0:  # 使用固定的10个epoch间隔，或者可以从配置中读取
            pretty_print_metrics(evaluation_results)
        
        # 日志记录
        logger.log({"train_loss": train_loss, "val_dice": overall_mean_dice}, step=epoch)

    logger.close()
    print("--- 微调任务完成 ---")
    print(f"最佳验证Dice: {best_val_dice:.4f}")
    print(f"最佳模型保存在: {best_model_path}")

if __name__ == '__main__':
    main_finetune() 