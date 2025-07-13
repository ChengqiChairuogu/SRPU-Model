import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from pathlib import Path
import sys
import time
from typing import Union

# --- 导入 ---
try:
    from configs import base as cfg_base
    from configs.train import train_config as cfg_train
    from configs import wandb_config as cfg_wandb
    from datasets.sem_datasets import SemSegmentationDataset
    from models.segmentation_unet import SegmentationUNet
    from utils.training_utils import DiceBCELoss, train_one_epoch, validate_one_epoch
except ImportError as e:
    print(f"导入模块时出错: {e}")
    if __name__ == '__main__' and __package__ is None:
        current_script_path = Path(__file__).resolve()
        project_root = current_script_path.parent.parent
        sys.path.insert(0, str(project_root))
        from configs import base as cfg_base
        from configs.train import train_config as cfg_train
        from configs import wandb_config as cfg_wandb
        from datasets.sem_datasets import SemSegmentationDataset
        from models.segmentation_unet import SegmentationUNet
        from utils.training_utils import DiceBCELoss, train_one_epoch, validate_one_epoch
    else:
        raise

def create_model(encoder_name: str, decoder_name: str, encoder_weights: Union[str, None]) -> nn.Module:
    """
    根据配置动态创建编码器和解码器，并组装成一个分割模型。
    """
    print(f"--- 正在创建模型: Encoder: {encoder_name}, Decoder: {decoder_name} ---")

    # 1. 动态选择并实例化编码器
    if encoder_name == 'unet':
        from models.encoders.unet_encoder import UNetEncoder
        encoder = UNetEncoder(in_channels=cfg_base.INPUT_DEPTH, base_c=64)
    # 在这里可以继续添加其他编码器的 'elif' 分支
    # elif encoder_name == 'resnet50':
    #     from models.encoders.resnet_encoder import ResNetEncoder
    #     encoder = ResNetEncoder(name='resnet50', in_channels=cfg_base.INPUT_DEPTH, weights=encoder_weights)
    else:
        raise ValueError(f"未知的编码器名称: '{encoder_name}'")

    # 2. 动态选择并实例化解码器
    encoder_channels = encoder.get_channels()
    if decoder_name == 'unet':
        from models.decoders.unet_decoder import UNetDecoder
        decoder = UNetDecoder(encoder_channels, num_classes=cfg_base.NUM_CLASSES)
    # 在这里可以继续添加其他解码器的 'elif' 分支
    # elif decoder_name == 'deeplab':
    #     from models.decoders.deeplab_decoder import DeepLabDecoder
    #     decoder = DeepLabDecoder(encoder_channels=encoder_channels, out_channels=cfg_base.NUM_CLASSES)
    else:
        raise ValueError(f"未知的解码器名称: '{decoder_name}'")

    # 3. 组装成最终的分割模型
    model = SegmentationUNet(encoder, decoder)
    return model

def main_train():
    project_root = cfg_base.PROJECT_ROOT.resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 开始监督学习训练任务: {cfg_train.TASK_NAME} ---")
    print(f"使用设备: {device}")

    # --- 初始化 wandb ---
    wandb.init(
        project=cfg_wandb.PROJECT_NAME_SUPERVISED,
        name=f"train-{cfg_train.ENCODER_NAME}-{cfg_train.DECODER_NAME}-{int(time.time())}",
        mode=cfg_wandb.WANDB_MODE,
        config={
            "task_name": cfg_train.TASK_NAME,
            "encoder": cfg_train.ENCODER_NAME,
            "decoder": cfg_train.DECODER_NAME,
            "encoder_weights": cfg_train.ENCODER_WEIGHTS,
            "learning_rate": cfg_train.LEARNING_RATE,
            "epochs": cfg_train.NUM_EPOCHS,
            "batch_size": cfg_train.BATCH_SIZE,
            "optimizer": cfg_train.OPTIMIZER,
        }
    )
    print(f"Wandb 在 '{wandb.run.settings.mode}' 模式下初始化。")

    # --- 数据加载 ---
    train_dataset = SemSegmentationDataset(
        json_file_identifier=cfg_train.TRAIN_JSON_NAME,
        project_root=project_root,
        is_train=True
    )
    val_dataset = SemSegmentationDataset(
        json_file_identifier=cfg_train.VAL_JSON_NAME,
        project_root=project_root,
        is_train=False
    )
    train_loader = DataLoader(
        train_dataset, batch_size=cfg_train.BATCH_SIZE,
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg_train.BATCH_SIZE * 2,
        shuffle=False, num_workers=4, pin_memory=True
    )

    # --- 模型与优化器创建 ---
    model = create_model(
        encoder_name=cfg_train.ENCODER_NAME,
        decoder_name=cfg_train.DECODER_NAME,
        encoder_weights=cfg_train.ENCODER_WEIGHTS
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg_train.LEARNING_RATE, weight_decay=cfg_train.WEIGHT_DECAY)
    criterion = DiceBCELoss()
    
    # --- 检查点目录与加载逻辑 ---
    checkpoint_dir = project_root / cfg_train.CHECKPOINT_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    start_epoch = 0
    best_metric = 0.0
    resumable_checkpoint_path = project_root / cfg_train.RESUMABLE_CHECKPOINT_PATH

    if cfg_train.RESUME_FROM_CHECKPOINT and resumable_checkpoint_path.exists():
        print(f"正在从检查点恢复训练: {resumable_checkpoint_path}")
        checkpoint = torch.load(resumable_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_metric = checkpoint.get('best_metric', 0.0)
        print(f"已从 epoch {start_epoch-1} 恢复。目前最佳Dice: {best_metric:.4f}。")
    else:
        print("从头开始训练。")
    
    if wandb.run.settings.mode != "disabled":
        wandb.watch(model, log="all", log_freq=100)

    # --- 训练循环 ---
    print(f"将从 Epoch {start_epoch + 1} 开始训练...")
    for epoch in range(start_epoch, cfg_train.NUM_EPOCHS):
        avg_train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        avg_val_loss, avg_val_dice = validate_one_epoch(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{cfg_train.NUM_EPOCHS} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}")

        is_best = avg_val_dice > best_metric
        if is_best:
            best_metric = avg_val_dice
            print(f"  -> 新的最佳Dice记录: {best_metric:.4f}。")

        # 保存最佳模型
        if is_best:
            best_model_path = project_root / cfg_train.BEST_MODEL_CHECKPOINT_PATH
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> 新的最佳模型已保存至 {best_model_path}")
            if wandb.run.settings.mode != "disabled":
                wandb.save(str(best_model_path))
        
        # 保存可恢复的检查点
        torch.save({
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'best_metric': best_metric,
        }, resumable_checkpoint_path)

        # 记录到W&B
        if wandb.run.settings.mode != "disabled":
            wandb.log({
                "epoch": epoch + 1, "train_loss": avg_train_loss,
                "val_loss": avg_val_loss, "val_dice": avg_val_dice,
                "best_val_dice": best_metric
            })

    if wandb.run.settings.mode != "disabled":
        wandb.finish()
    print("训练结束。")

if __name__ == '__main__':
    main_train()