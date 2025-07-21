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
from models.unet_autoencoder import UNetAutoencoder
import matplotlib.pyplot as plt
import os
from utils.training_utils import SSIMLoss, VGGPerceptualLoss
import random
from utils.logger import Logger

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

SEED = getattr(cfg_ssl, 'SEED', 42)  # 从config读取全局随机种子
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# 若后续有albumentations官方set_seed方法可用，可补充

def create_encoder(encoder_name: str) -> nn.Module:
    """
    根据配置动态创建编码器实例。
    """
    print(f"--- 正在创建编码器: {encoder_name} ---")
    
    if encoder_name == 'unet':
        from models.encoders.unet_encoder import UNetEncoder
        encoder = UNetEncoder(n_channels=1)
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
                rec_img = np.squeeze(reconstructed_images[i])  # 修正变量名
                data_range = ori_img.max() - ori_img.min()
                if data_range == 0: data_range = 1
                psnr_val = psnr(ori_img, rec_img, data_range=data_range)
                if isinstance(psnr_val, tuple):
                    psnr_val = psnr_val[0]
                total_psnr += float(psnr_val)
                try:
                    ssim_val = ssim(ori_img, rec_img, data_range=data_range)
                    if isinstance(ssim_val, tuple):
                        ssim_val = ssim_val[0]
                    total_ssim += float(ssim_val)
                except ValueError:
                    pass
            num_samples += original_images_cpu.shape[0]
    avg_psnr = total_psnr / num_samples if num_samples > 0 else 0
    avg_ssim = total_ssim / num_samples if num_samples > 0 else 0
    return avg_psnr, avg_ssim

def visualize_ssl_batch(masked_images, original_images, reconstructed_images, loss_masks, save_dir, prefix="epoch", max_samples=8):
    os.makedirs(save_dir, exist_ok=True)
    masked_images = masked_images.cpu().numpy()
    original_images = original_images.cpu().numpy()
    reconstructed_images = reconstructed_images.cpu().numpy()
    loss_masks = loss_masks.cpu().numpy()
    n = min(max_samples, masked_images.shape[0])
    for i in range(n):
        fig, axs = plt.subplots(1, 4, figsize=(12, 3))
        axs[0].imshow(original_images[i][0], cmap="gray")
        axs[0].set_title("Original")
        axs[1].imshow(masked_images[i][0], cmap="gray")
        axs[1].set_title("Masked Input")
        axs[2].imshow(reconstructed_images[i][0], cmap="gray")
        axs[2].set_title("Reconstructed")
        axs[3].imshow(loss_masks[i], cmap="gray")
        axs[3].set_title("Mask Area")
        for ax in axs:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{prefix}_sample_{i}.png"))
        plt.close(fig)

def train_ssl_mae():
    project_root = cfg_base.PROJECT_ROOT.resolve()
    # 修复设备检测
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA 可用，使用 GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("CUDA 不可用，使用 CPU")
    
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
            # "patch_size": cfg_ssl.PATCH_SIZE,  # 已删除
            "mask_ratio": cfg_ssl.MASK_RATIO,
        }
    )
    if wandb.run is not None:
        print(f"Wandb 在 '{getattr(wandb.run, 'mode', 'unknown')}' 模式下初始化。")
    else:
        print("Wandb 未启用或初始化失败。")

    json_path = project_root / cfg_ssl.JSON_DIR_NAME / cfg_ssl.UNLABELED_JSON_NAME
    ssl_dataset = SSLDataset(json_file_path=json_path, project_root=project_root)
    ssl_loader = DataLoader(ssl_dataset, batch_size=cfg_ssl.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    vis_loader = DataLoader(ssl_dataset, batch_size=8, shuffle=True)
    vis_batch = next(iter(vis_loader)) if len(vis_loader) > 0 else None

    # encoder = create_encoder(cfg_ssl.ENCODER_NAME)
    # model = MaskedAutoencoderUNet(
    #     encoder=encoder,
    #     decoder_embed_dim=cfg_ssl.DECODER_EMBED_DIM,
    #     n_channels_in=1
    # ).to(device)
    model = UNetAutoencoder(n_channels=1, n_classes=1).to(device)
    
    # 损失函数组合：MSE + SSIM + 感知
    mse_criterion = nn.MSELoss(reduction='none')
    ssim_criterion = SSIMLoss(window_size=11)
    perceptual_criterion = VGGPerceptualLoss(resize=True)
    def combined_ssl_loss(pred, target, mask):
        # mask: (B, H, W) or (B, 1, H, W)
        mask_exp = mask.unsqueeze(1) if mask.dim() == 3 else mask
        mse = (mse_criterion(pred, target) * mask_exp).sum() / mask_exp.sum().clamp(min=1e-5)
        # SSIM和感知损失不加mask（可选加权）
        ssim = ssim_criterion(pred, target)
        perceptual = perceptual_criterion(pred, target)
        return mse + 0.5 * ssim + 0.1 * perceptual
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg_ssl.LEARNING_RATE, weight_decay=cfg_ssl.WEIGHT_DECAY)
    
    if wandb.run is not None and getattr(wandb.run, 'mode', None) != "disabled":
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
    logger = Logger(cfg_ssl.log_config)
    for epoch in range(start_epoch, cfg_ssl.NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(ssl_loader, desc=f"Epoch {epoch+1}/{cfg_ssl.NUM_EPOCHS}", leave=False)
        for masked_images, original_images, loss_masks in progress_bar:
            masked_images, original_images, loss_masks = masked_images.to(device), original_images.to(device), loss_masks.to(device)
            optimizer.zero_grad()
            reconstructed_images = model(masked_images)
            #loss_masks_expanded = loss_masks.unsqueeze(1)
            #per_pixel_loss = criterion(reconstructed_images, original_images)
            #loss = (per_pixel_loss * loss_masks_expanded).sum() / loss_masks_expanded.sum().clamp(min=1e-5)
            loss = combined_ssl_loss(reconstructed_images, original_images, loss_masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        # --- 可视化调试: 每个epoch保存一批样本 ---
        if (epoch + 1) % 1 == 0:  # 可按需调整频率
            model.eval()
            with torch.no_grad():
                vis_masked, vis_original, vis_loss_mask = next(iter(vis_loader))
                vis_masked = vis_masked.to(device)
                vis_original = vis_original.to(device)
                vis_loss_mask = vis_loss_mask.to(device)
                vis_reconstructed = model(vis_masked)
                visualize_ssl_batch(
                    vis_masked, vis_original, vis_reconstructed, vis_loss_mask,
                    save_dir=str(project_root / "data" / "ssl_debug_vis"),
                    prefix=f"epoch{epoch+1}", max_samples=8
                )
            model.train()

        avg_loss = total_loss / len(ssl_loader)
        # === 新增：每个epoch都评估PSNR和SSIM ===
        avg_psnr, avg_ssim = evaluate_reconstruction_metrics(model, vis_loader, device)
        print(f"Epoch [{epoch+1}/{cfg_ssl.NUM_EPOCHS}], 平均重建损失: {avg_loss:.6f}, Validation PSNR: {avg_psnr:.4f} dB, SSIM: {avg_ssim:.4f}")

        log_dict = {
            "epoch": epoch + 1,
            "ssl_loss": avg_loss,
            "val_psnr": avg_psnr,
            "val_ssim": avg_ssim
        }
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
            print(f"  -> 新的最佳损失记录: {best_loss:.6f}。")

        if (epoch + 1) % cfg_ssl.SAVE_BEST_CHECK_EVERY_N_EPOCHS == 0 or (epoch + 1) == cfg_ssl.NUM_EPOCHS:
            if is_best:
                best_model_path = project_root / cfg_ssl.BEST_MODEL_CHECKPOINT_PATH
                print(f"  -> 正在保存 epoch {epoch+1} 的最佳模型编码器至 {best_model_path}...")
                torch.save(model.encoder.state_dict(), best_model_path)
                if wandb.run is not None and getattr(wandb.run, 'mode', None) != "disabled":
                    wandb.save(str(best_model_path)) 
            
            if vis_batch and wandb.run is not None and getattr(wandb.run, 'mode', None) != "disabled":
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
        
        if wandb.run is not None and getattr(wandb.run, 'mode', None) != "disabled":
            wandb.log(log_dict)
        logger.log(log_dict, step=epoch+1)
    logger.close()

    final_encoder_path = project_root / cfg_ssl.SSL_ENCODER_FINAL_PATH
    torch.save(model.encoder.state_dict(), final_encoder_path)
    if wandb.run is not None and getattr(wandb.run, 'mode', None) != "disabled":
        wandb.save(str(final_encoder_path))
        wandb.finish()
    
    print(f"训练结束。最终SSL预训练编码器已保存至 {final_encoder_path}")

    # 新增：训练-评估对比可视化（与inspect一致）
    compare_dir = project_root / "data" / "ssl_compare"
    compare_dir.mkdir(parents=True, exist_ok=True)
    compare_idx = 290
    print(f"[对比模式] 训练脚本保存索引 {compare_idx} 的可视化结果")
    masked_img, original_img, loss_mask = ssl_dataset[compare_idx]
    masked_img_tensor = masked_img.unsqueeze(0).to(device)
    with torch.no_grad():
        recon_img = model(masked_img_tensor).cpu().numpy()[0, 0]
    import matplotlib.pyplot as plt
    plt.imsave(compare_dir / "train_masked.png", masked_img[0], cmap="gray")
    plt.imsave(compare_dir / "train_mask.png", loss_mask, cmap="gray")
    plt.imsave(compare_dir / "train_recon.png", recon_img, cmap="gray")
    plt.imsave(compare_dir / "train_original.png", original_img[0], cmap="gray")
    print(f"[对比模式] 已保存训练Masked Input、Mask Area、Reconstructed、Original到: {compare_dir}")

if __name__ == '__main__':
    train_ssl_mae()