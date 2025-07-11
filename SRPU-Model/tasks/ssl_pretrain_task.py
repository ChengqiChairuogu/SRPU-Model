import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import sys
import time
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from typing import Optional, List, Tuple, Dict, Any

# --- 配置与模块导入 ---
try:
    from configs import base as cfg_base
    from configs import json_config as cfg_json_gen
    from configs.selfup import ssl_config as cfg_ssl
    from datasets.ssl_dataset import SSLDataset
    # 导入您自己的、模块化的模型
    from models.encoders.unet_encoder import UNetEncoder
    from models.mae_model import MaskedAutoencoderUNet

except ImportError:
    # ... (健壮的导入逻辑，以支持直接运行脚本) ...
    if __name__ == '__main__' and __package__ is None:
        current_script_path = Path(__file__).resolve()
        project_root = current_script_path.parent.parent
        sys.path.insert(0, str(project_root))
        try:
            from configs import base as cfg_base
            from configs import json_config as cfg_json_gen
            from configs.selfup import ssl_config as cfg_ssl
            from datasets.ssl_dataset import SSLDataset
            from models.encoders.unet_encoder import UNetEncoder
            from models.mae_model import MaskedAutoencoderUNet
        except ImportError as e_inner:
            print(f"Error importing modules in ssl_pretrain_task.py: {e_inner}")
            print("Please ensure all required files (configs, datasets, models) exist.")
            sys.exit(1)
    else:
        raise

# --- 可视化与评估函数 ---
def visualize_reconstruction(original_imgs, masked_imgs, reconstructed_imgs, loss_masks, save_path, epoch, num_images=8):
    """
    将原始图像、遮挡图像、模型重建和补齐后的完整图像并排保存。
    """
    num_images = min(num_images, original_imgs.size(0))
    original_sample = original_imgs[:num_images]
    masked_sample = masked_imgs[:num_images]
    reconstructed_sample = reconstructed_imgs[:num_images]
    loss_masks_sample = loss_masks[:num_images].unsqueeze(1).float()
    completed_sample = original_sample * (1 - loss_masks_sample) + reconstructed_sample * loss_masks_sample
    comparison = torch.cat([original_sample, masked_sample, reconstructed_sample, completed_sample], dim=0)
    save_image(comparison.cpu(), save_path, nrow=num_images, normalize=True)
    print(f"Epoch {epoch}: 可视化结果已保存到 {save_path}")

def evaluate_reconstruction_metrics(model, dataloader, device):
    """
    在给定的数据集上评估模型的重建性能 (PSNR, SSIM)。
    """
    model.eval()
    total_psnr, total_ssim, num_samples = 0.0, 0.0, 0
    with torch.no_grad():
        for masked_images, original_images, _ in dataloader:
            masked_images = masked_images.to(device)
            # 为了节约GPU显存，计算指标时可以在CPU上进行
            original_images_cpu = original_images.numpy()
            reconstructed_images_cpu = model(masked_images).cpu().numpy()
            for i in range(original_images_cpu.shape[0]):
                ori_img = np.squeeze(original_images_cpu[i])
                rec_img = np.squeeze(reconstructed_images_cpu[i])
                data_range = ori_img.max() - ori_img.min()
                if data_range == 0: data_range = 1
                total_psnr += psnr(ori_img, rec_img, data_range=data_range)
                # For grayscale, channel_axis is deprecated in newer scikit-image, use multichannel=False if needed
                # For now, let's assume it works or adapt to ssim(..., multichannel=False) for newer versions
                total_ssim += ssim(ori_img, rec_img, data_range=data_range)
            num_samples += original_images_cpu.shape[0]
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    return avg_psnr, avg_ssim


# --- 主训练函数 ---
def train_ssl_mae():
    project_root = cfg_base.PROJECT_ROOT.resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. 数据集和加载器 ---
    ssl_transform = transforms.Compose([transforms.Resize((cfg_base.IMAGE_HEIGHT, cfg_base.IMAGE_WIDTH)), transforms.ToTensor()])
    json_path = project_root / cfg_json_gen.JSON_OUTPUT_DIR_NAME / cfg_ssl.UNLABELED_JSON_NAME
    ssl_dataset = SSLDataset(json_file_path=json_path, project_root=project_root, raw_image_root_name_in_json=cfg_json_gen.RAW_IMAGE_SOURCE_DIR_NAME, patch_size=cfg_ssl.PATCH_SIZE, mask_ratio=cfg_ssl.MASK_RATIO, transform=ssl_transform)
    ssl_loader = DataLoader(ssl_dataset, batch_size=cfg_ssl.BATCH_SIZE, shuffle=True, num_workers=cfg_ssl.NUM_WORKERS, pin_memory=True, drop_last=True)
    try:
        vis_loader = DataLoader(ssl_dataset, batch_size=8, shuffle=False)
        vis_batch = next(iter(vis_loader))
    except StopIteration:
        vis_batch = None
        print("Warning: SSL dataset is empty or too small for a visualization batch.")
    if len(ssl_dataset) == 0: print("Dataset is empty. Cannot start training."); return

    # --- 2. 模型、损失函数、优化器 ---
    print("Initializing models...")
    encoder = UNetEncoder(in_channels=1, base_c=64)
    model = MaskedAutoencoderUNet(encoder=encoder, decoder_embed_dim=cfg_ssl.DECODER_EMBED_DIM, n_channels_in=1).to(device)
    criterion = nn.MSELoss(reduction='none')
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg_ssl.LEARNING_RATE, weight_decay=cfg_ssl.WEIGHT_DECAY)
    
    # --- 3. 日志与检查点路径的初始化 ---
    final_encoder_save_path = project_root / cfg_ssl.SSL_ENCODER_FINAL_PATH
    best_encoder_save_path = project_root / cfg_ssl.SSL_BEST_MODEL_CHECKPOINT_PATH
    resumable_checkpoint_path = project_root / cfg_ssl.SSL_RESUMABLE_CHECKPOINT_PATH
    final_encoder_save_path.parent.mkdir(parents=True, exist_ok=True)
    visualization_dir = project_root / cfg_ssl.VISUALIZATION_DIR
    visualization_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = project_root / "logs" / cfg_ssl.LOG_DIR_NAME / f"{cfg_ssl.MODEL_NAME}_{int(time.time())}"
    
    # --- 4. 加载检查点以恢复训练 ---
    start_epoch = 0
    best_loss = float('inf') 

    if cfg_ssl.RESUME_FROM_CHECKPOINT:
        resume_path = project_root / cfg_ssl.RESUME_FROM_CHECKPOINT
        if resume_path.exists():
            print(f"Resuming training from checkpoint: {resume_path}")
            try:
                checkpoint = torch.load(resume_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_loss = checkpoint.get('best_loss', float('inf')) 
                
                if 'log_dir' in checkpoint and Path(checkpoint['log_dir']).is_dir():
                    log_dir = Path(checkpoint['log_dir'])
                    print(f"Logs will continue in existing directory: {log_dir}")
                
                print(f"Resumed from epoch {start_epoch-1}. Best loss so far: {best_loss:.6f}. Starting next epoch: {start_epoch}.")
            except Exception as e:
                print(f"Error loading checkpoint: {e}. Starting training from scratch.")
                start_epoch = 0
                best_loss = float('inf')
        else:
            print(f"Warning: Checkpoint file not found at {resume_path}. Starting training from scratch.")
    else:
        print("Starting training from scratch.")

    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard logs will be saved to: {log_dir}")
    print(f"Final encoder will be saved to: {final_encoder_save_path}")
    print(f"Best encoder (by loss) will be saved to: {best_encoder_save_path}")
    print(f"Resumable checkpoints will be saved to: {resumable_checkpoint_path}")

    # --- 5. 训练循环 ---
    print(f"Starting Self-Supervised Learning. Target Epochs: {cfg_ssl.NUM_EPOCHS}")
    for epoch in range(start_epoch, cfg_ssl.NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        for i, (masked_images, original_images, loss_masks) in enumerate(ssl_loader):
            masked_images, original_images, loss_masks = masked_images.to(device), original_images.to(device), loss_masks.to(device)
            optimizer.zero_grad()
            reconstructed_images = model(masked_images)
            loss_masks_expanded = loss_masks.unsqueeze(1)
            per_pixel_loss = criterion(reconstructed_images, original_images)
            loss = (per_pixel_loss * loss_masks_expanded).sum() / loss_masks_expanded.sum().clamp(min=1e-5)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(ssl_loader)
        writer.add_scalar('Loss/SSL_Reconstruction', avg_loss, epoch + 1)
        print(f"Epoch [{epoch+1}/{cfg_ssl.NUM_EPOCHS}], Average Reconstruction Loss: {avg_loss:.6f}")

        # --- 修改部分：保存最佳模型 ---
        # 1. 无论何时，只要损失降低，就更新 best_loss 变量
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
            print(f"  -> New best loss recorded: {best_loss:.6f}.")

        # 2. 定期检查是否要保存最佳模型 (也包括最后一个周期)
        if (epoch + 1) % cfg_ssl.SAVE_BEST_CHECK_EVERY_N_EPOCHS == 0 or (epoch + 1) == cfg_ssl.NUM_EPOCHS:
            if is_best:
                print(f"  -> Saving new best model encoder at epoch {epoch+1}...")
                torch.save(model.encoder.state_dict(), best_encoder_save_path)
            else:
                print(f"  -> Loss did not improve at this check point. Best loss remains {best_loss:.6f}.")

        # 定期进行可视化和量化评估
        if (epoch + 1) % cfg_ssl.VISUALIZE_EVERY_N_EPOCHS == 0 or (epoch + 1) == cfg_ssl.NUM_EPOCHS:
            model.eval()
            if vis_batch:
                with torch.no_grad():
                    vis_masked, vis_original, vis_loss_masks = vis_batch
                    vis_masked_gpu = vis_masked.to(device)
                    vis_reconstructed = model(vis_masked_gpu)
                save_path = visualization_dir / f"reconstruction_epoch_{epoch+1:04d}.png"
                visualize_reconstruction(vis_original, vis_masked, vis_reconstructed.cpu(), vis_loss_masks, save_path, epoch + 1)
            
            avg_psnr, avg_ssim = evaluate_reconstruction_metrics(model, vis_loader, device)
            writer.add_scalar('Metrics/PSNR', avg_psnr, epoch + 1)
            writer.add_scalar('Metrics/SSIM', avg_ssim, epoch + 1)
            print(f"Epoch [{epoch+1}/{cfg_ssl.NUM_EPOCHS}], Validation PSNR: {avg_psnr:.4f} dB, SSIM: {avg_ssim:.4f}")
            model.train() # 评估结束后切回训练模式

        # 每个周期结束时都保存可恢复的检查点
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'best_loss': best_loss, # <-- 保存当前最佳损失
            'log_dir': str(log_dir)
        }, resumable_checkpoint_path)

    # 训练结束后，保存最终的编码器权重
    torch.save(model.encoder.state_dict(), final_encoder_save_path)
    print(f"Training finished. Final SSL-pretrained encoder saved to {final_encoder_save_path}")
    writer.close()

if __name__ == '__main__':
    train_ssl_mae()
