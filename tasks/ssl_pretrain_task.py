import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from pathlib import Path
import sys
import time
from torchvision.utils import make_grid
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from typing import Tuple, Union
from tqdm import tqdm # **关键修正**: 导入tqdm库

# --- 配置与模块导入 ---
try:
    from configs import base as cfg_base
    from configs.selfup import ssl_config as cfg_ssl
    from configs import wandb_config as cfg_wandb
    from datasets.ssl_dataset import SSLDataset
    from models.mae_model import MaskedAutoencoderUNet
except ImportError as e:
    print(f"导入模块时出错: {e}")
    if __name__ == '__main__' and __package__ is None:
        current_script_path = Path(__file__).resolve()
        project_root = current_script_path.parent.parent
        sys.path.insert(0, str(project_root))
        from configs import base as cfg_base
        from configs.selfup import ssl_config as cfg_ssl
        from configs import wandb_config as cfg_wandb
        from datasets.ssl_dataset import SSLDataset
        from models.mae_model import MaskedAutoencoderUNet
    else:
        raise

def create_encoder(encoder_name: str) -> nn.Module:
    """
    根据配置动态创建编码器实例。
    """
    print(f"--- 正在创建编码器: {encoder_name} ---")
    
    if encoder_name == 'unet':
        from models.encoders.unet_encoder import UNetEncoder
        encoder = UNetEncoder(in_channels=1, base_c=64)
    # 在这里可以继续添加其他编码器的 'elif' 分支
    # elif encoder_name == 'resnet50':
    #     from models.encoders.resnet_encoder import ResNetEncoder
    #     encoder = ResNetEncoder(name='resnet50', in_channels=1, weights=None)
    else:
        raise ValueError(f"未知的编码器名称: '{encoder_name}'")
    return encoder

def evaluate_reconstruction_metrics(model, dataloader, device) -> Tuple[float, float]:
    model.eval()
    total_psnr, total_ssim, num_samples = 0.0, 0.0, 0
    with torch.no_grad():
        for _, original_images, _ in dataloader:
            masked_images = original_images.to(device)
            original_images_cpu = original_images.numpy()
            reconstructed_images = model(masked_images).cpu().numpy()
            for i in range(original_images_cpu.shape[0]):
                ori_img = np.squeeze(original_images_cpu[i])
                rec_img = np.squeeze(reconstructed_images_cpu[i])
                data_range = ori_img.max() - ori_img.min()
                if data_range == 0: data_range = 1
                total_psnr += psnr(ori_img, rec_img, data_range=data_range)
                try:
                    ssim_val = ssim(ori_img, rec_img, data_range=data_range)
                    total_ssim += ssim_val
                except ValueError:
                    pass
            num_samples += original_images_cpu.shape[0]
    avg_psnr = total_psnr / num_samples if num_samples > 0 else 0
    avg_ssim = total_ssim / num_samples if num_samples > 0 else 0
    return avg_psnr, avg_ssim

def train_ssl_mae():
    project_root = cfg_base.PROJECT_ROOT.resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 开始自监督预训练任务: {cfg_ssl.MODEL_NAME} ---")
    print(f"使用设备: {device}")

    wandb.init(
        project=cfg_wandb.PROJECT_NAME_SSL,
        name=f"ssl-{cfg_ssl.MODEL_NAME}-{int(time.time())}",
        mode=cfg_wandb.WANDB_MODE,
        config={
            "encoder": cfg_ssl.ENCODER_NAME,
            "learning_rate": cfg_ssl.LEARNING_RATE,
            "epochs": cfg_ssl.NUM_EPOCHS,
            "batch_size": cfg_ssl.BATCH_SIZE,
            "patch_size": cfg_ssl.PATCH_SIZE,
            "mask_ratio": cfg_ssl.MASK_RATIO,
        }
    )
    print(f"Wandb 在 '{wandb.run.settings.mode}' 模式下初始化。")

    json_path = project_root / cfg_ssl.JSON_DIR_NAME / cfg_ssl.UNLABELED_JSON_NAME
    ssl_dataset = SSLDataset(json_file_path=json_path, project_root=project_root)
    ssl_loader = DataLoader(ssl_dataset, batch_size=cfg_ssl.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    vis_loader = DataLoader(ssl_dataset, batch_size=8, shuffle=True)
    vis_batch = next(iter(vis_loader)) if len(vis_loader) > 0 else None

    encoder = create_encoder(cfg_ssl.ENCODER_NAME)
    model = MaskedAutoencoderUNet(
        encoder=encoder,
        decoder_embed_dim=cfg_ssl.DECODER_EMBED_DIM,
        n_channels_in=1
    ).to(device)
    
    criterion = nn.MSELoss(reduction='none')
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg_ssl.LEARNING_RATE, weight_decay=cfg_ssl.WEIGHT_DECAY)
    
    if wandb.run.settings.mode != "disabled":
        wandb.watch(model, log="all", log_freq=100)
    
    checkpoint_dir = project_root / cfg_ssl.SSL_CHECKPOINT_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    start_epoch = 0
    best_loss = float('inf') 
    resumable_checkpoint_path = project_root / cfg_ssl.RESUMABLE_CHECKPOINT_PATH
    if cfg_ssl.RESUME_FROM_CHECKPOINT and resumable_checkpoint_path.exists():
        print(f"正在从检查点恢复训练: {resumable_checkpoint_path}")
        checkpoint = torch.load(resumable_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"已从 epoch {start_epoch-1} 恢复。目前最佳损失: {best_loss:.6f}。")
    else:
        print("从头开始训练。")

    print(f"将从 Epoch {start_epoch + 1} 开始训练...")
    for epoch in range(start_epoch, cfg_ssl.NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(ssl_loader, desc=f"Epoch {epoch+1}/{cfg_ssl.NUM_EPOCHS}", leave=False)
        for masked_images, original_images, loss_masks in progress_bar:
            masked_images, original_images, loss_masks = masked_images.to(device), original_images.to(device), loss_masks.to(device)
            optimizer.zero_grad()
            reconstructed_images = model(masked_images)
            loss_masks_expanded = loss_masks.unsqueeze(1)
            per_pixel_loss = criterion(reconstructed_images, original_images)
            loss = (per_pixel_loss * loss_masks_expanded).sum() / loss_masks_expanded.sum().clamp(min=1e-5)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(ssl_loader)
        print(f"Epoch [{epoch+1}/{cfg_ssl.NUM_EPOCHS}], 平均重建损失: {avg_loss:.6f}")
        
        log_dict = {"epoch": epoch + 1, "ssl_loss": avg_loss}
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
            print(f"  -> 新的最佳损失记录: {best_loss:.6f}。")

        if (epoch + 1) % cfg_ssl.SAVE_BEST_CHECK_EVERY_N_EPOCHS == 0 or (epoch + 1) == cfg_ssl.NUM_EPOCHS:
            if is_best:
                best_model_path = project_root / cfg_ssl.BEST_MODEL_CHECKPOINT_PATH
                print(f"  -> 正在保存 epoch {epoch+1} 的最佳模型编码器至 {best_model_path}...")
                torch.save(model.encoder.state_dict(), best_model_path)
                if wandb.run.settings.mode != "disabled":
                    wandb.save(str(best_model_path)) 
            
            if vis_batch and wandb.run.settings.mode != "disabled":
                model.eval()
                with torch.no_grad():
                    vis_masked, vis_original, _ = vis_batch
                    vis_reconstructed = model(vis_masked.to(device)).cpu()
                    grid_original = make_grid(vis_original, nrow=4, normalize=True)
                    grid_masked = make_grid(vis_masked, nrow=4, normalize=True)
                    grid_reconstructed = make_grid(vis_reconstructed, nrow=4, normalize=True)
                    log_dict["visualizations/original"] = wandb.Image(grid_original)
                    log_dict["visualizations/masked"] = wandb.Image(grid_masked)
                    log_dict["visualizations/reconstructed"] = wandb.Image(grid_reconstructed)
                
                avg_psnr, avg_ssim = evaluate_reconstruction_metrics(model, vis_loader, device)
                log_dict["val_psnr"] = avg_psnr
                log_dict["val_ssim"] = avg_ssim
                print(f"Epoch [{epoch+1}/{cfg_ssl.NUM_EPOCHS}], Validation PSNR: {avg_psnr:.4f} dB, SSIM: {avg_ssim:.4f}")
                model.train()

        torch.save({
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'loss': avg_loss, 'best_loss': best_loss,
        }, resumable_checkpoint_path)
        
        if wandb.run.settings.mode != "disabled":
            wandb.log(log_dict)

    final_encoder_path = project_root / cfg_ssl.SSL_ENCODER_FINAL_PATH
    torch.save(model.encoder.state_dict(), final_encoder_path)
    if wandb.run.settings.mode != "disabled":
        wandb.save(str(final_encoder_path))
        wandb.finish()
    
    print(f"训练结束。最终SSL预训练编码器已保存至 {final_encoder_path}")

if __name__ == '__main__':
    train_ssl_mae()