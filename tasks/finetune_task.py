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
    from configs.finetune import finetune_config as cfg_finetune
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
        from configs.finetune import finetune_config as cfg_finetune
        from configs import wandb_config as cfg_wandb
        from datasets.sem_datasets import SemSegmentationDataset
        from models.segmentation_unet import SegmentationUNet
        from utils.training_utils import DiceBCELoss, train_one_epoch, validate_one_epoch
    else:
        raise

def create_model(encoder_name: str, decoder_name: str) -> nn.Module:
    """
    根据配置动态创建编码器和解码器。
    """
    print(f"--- 正在创建模型: Encoder: {encoder_name}, Decoder: {decoder_name} ---")

    if encoder_name == 'unet':
        from models.encoders.unet_encoder import UNetEncoder
        encoder = UNetEncoder(in_channels=cfg_base.INPUT_DEPTH, base_c=64)
    else:
        raise ValueError(f"未知的编码器名称: '{encoder_name}'")

    encoder_channels = encoder.get_channels()
    if decoder_name == 'unet':
        from models.decoders.unet_decoder import UNetDecoder
        decoder = UNetDecoder(encoder_channels, num_classes=cfg_base.NUM_CLASSES)
    else:
        raise ValueError(f"未知的解码器名称: '{decoder_name}'")

    model = SegmentationUNet(encoder, decoder)
    return model

def main_finetune():
    project_root = cfg_base.PROJECT_ROOT.resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 开始微调训练任务: {cfg_finetune.TASK_NAME} ---")
    print(f"使用设备: {device}")
    print(f"微调模式: {cfg_finetune.FINETUNE_MODE}")

    # --- 初始化 wandb ---
    wandb.init(
        project=cfg_wandb.PROJECT_NAME_SUPERVISED,
        name=f"finetune-{cfg_finetune.ENCODER_NAME}-{cfg_finetune.DECODER_NAME}-{int(time.time())}",
        mode=cfg_wandb.WANDB_MODE,
        config={
            "task_name": cfg_finetune.TASK_NAME,
            "encoder": cfg_finetune.ENCODER_NAME,
            "decoder": cfg_finetune.DECODER_NAME,
            "finetune_mode": cfg_finetune.FINETUNE_MODE,
            "base_learning_rate": cfg_finetune.BASE_LEARNING_RATE,
            "encoder_learning_rate": cfg_finetune.ENCODER_LEARNING_RATE,
            "epochs": cfg_finetune.NUM_EPOCHS,
            "batch_size": cfg_finetune.BATCH_SIZE,
        }
    )
    print(f"Wandb 在 '{wandb.run.settings.mode}' 模式下初始化。")

    # --- 数据加载 ---
    train_dataset = SemSegmentationDataset(
        json_file_identifier=cfg_finetune.TRAIN_JSON_NAME,
        project_root=project_root, is_train=True
    )
    val_dataset = SemSegmentationDataset(
        json_file_identifier=cfg_finetune.VAL_JSON_NAME,
        project_root=project_root, is_train=False
    )
    train_loader = DataLoader(
        train_dataset, batch_size=cfg_finetune.BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg_finetune.BATCH_SIZE * 2, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # --- 模型创建与权重加载 ---
    model = create_model(
        encoder_name=cfg_finetune.ENCODER_NAME,
        decoder_name=cfg_finetune.DECODER_NAME
    ).to(device)

    pretrained_path = project_root / cfg_finetune.PRETRAINED_ENCODER_PATH
    if pretrained_path.exists():
        print(f"正在从以下路径加载SSL预训练编码器权重: {pretrained_path}")
        try:
            encoder_state_dict = torch.load(pretrained_path, map_location=device)
            model.encoder.load_state_dict(encoder_state_dict, strict=True)
            print("SSL预训练编码器权重加载成功。")
        except Exception as e:
            print(f"加载SSL预训练权重时出错: {e}。")
    else:
        print(f"警告: 在 {pretrained_path} 未找到SSL预训练编码器。")
    
    # --- 优化器设置 ---
    if cfg_finetune.FINETUNE_MODE == 'finetune_frozen':
        print("优化器设置: 冻结编码器，只训练解码器。")
        for param in model.encoder.parameters():
            param.requires_grad = False
        params_to_update = model.decoder.parameters()
        optimizer = torch.optim.AdamW(params_to_update, lr=cfg_finetune.BASE_LEARNING_RATE, weight_decay=cfg_finetune.WEIGHT_DECAY)
    elif cfg_finetune.FINETUNE_MODE == 'finetune_differential':
        print("优化器设置: 使用差分学习率。")
        optimizer = torch.optim.AdamW([
            {'params': model.encoder.parameters(), 'lr': cfg_finetune.ENCODER_LEARNING_RATE},
            {'params': model.decoder.parameters(), 'lr': cfg_finetune.BASE_LEARNING_RATE}
        ], weight_decay=cfg_finetune.WEIGHT_DECAY)
    else: # 'finetune_full'
        print("优化器设置: 使用基础学习率训练所有参数。")
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg_finetune.BASE_LEARNING_RATE, weight_decay=cfg_finetune.WEIGHT_DECAY)

    criterion = DiceBCELoss()
    
    # --- 检查点目录与加载逻辑 ---
    checkpoint_dir = project_root / cfg_finetune.CHECKPOINT_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best_metric = 0.0
    resumable_checkpoint_path = project_root / cfg_finetune.RESUMABLE_CHECKPOINT_PATH

    if cfg_finetune.RESUME_FROM_CHECKPOINT and resumable_checkpoint_path.exists():
        print(f"正在从检查点恢复训练: {resumable_checkpoint_path}")
        checkpoint = torch.load(resumable_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_metric = checkpoint.get('best_metric', 0.0)
        print(f"已从 epoch {start_epoch-1} 恢复。目前最佳Dice: {best_metric:.4f}。")
    else:
        print("从头开始微调 (或从SSL预训练权重开始)。")
    
    if wandb.run.settings.mode != "disabled":
        wandb.watch(model, log="all", log_freq=100)

    # --- 训练循环 ---
    print(f"将从 Epoch {start_epoch + 1} 开始训练...")
    for epoch in range(start_epoch, cfg_finetune.NUM_EPOCHS):
        avg_train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        avg_val_loss, avg_val_dice = validate_one_epoch(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{cfg_finetune.NUM_EPOCHS} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}")
        
        is_best = avg_val_dice > best_metric
        if is_best:
            best_metric = avg_val_dice
            print(f"  -> 新的最佳Dice记录: {best_metric:.4f}。")

        # 保存最佳模型
        if is_best:
            best_model_path = project_root / cfg_finetune.BEST_MODEL_CHECKPOINT_PATH
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
    print("微调结束。")

if __name__ == '__main__':
    main_finetune()