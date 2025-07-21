import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import sys
import random

# --- 配置与模块导入 ---
try:
    from configs import base as cfg_base
    from configs.selfup import ssl_inspect_config as cfg_eval
    from datasets.ssl_dataset import SSLDataset
    from models.unet_autoencoder import UNetAutoencoder
except ImportError as e:
    print(f"导入模块时出错: {e}")
    if __name__ == '__main__' and __package__ is None:
        current_script_path = Path(__file__).resolve()
        project_root = current_script_path.parent.parent
        sys.path.insert(0, str(project_root))
        from configs import base as cfg_base
        from configs.selfup import ssl_inspect_config as cfg_eval
        from datasets.ssl_dataset import SSLDataset
        from models.unet_autoencoder import UNetAutoencoder
    else:
        raise

# 1. 同步全局随机种子
SEED = getattr(cfg_eval, 'SEED', 42)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 2. 读取对比图片索引
compare_idx = getattr(cfg_eval, 'COMPARE_INDEX', 0)

# 3. 路径与设备
project_root = cfg_base.PROJECT_ROOT.resolve()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 4. 加载数据集（单线程，保证一致性）
json_path = project_root / cfg_eval.JSON_DIR_NAME / cfg_eval.UNLABELED_JSON_NAME
ssl_dataset = SSLDataset(json_file_path=json_path, project_root=project_root)

# 5. 直接获取指定索引的图片
masked_img, original_img, loss_mask = ssl_dataset[compare_idx]
masked_img_tensor = masked_img.unsqueeze(0).to(device)

# 6. 加载模型与权重
model = UNetAutoencoder(n_channels=1, n_classes=1).to(device)
checkpoint_path = project_root / cfg_eval.MODEL_CHECKPOINT_PATH
if checkpoint_path.exists():
    checkpoint = torch.load(checkpoint_path, map_location=device)
    try:
        if 'model_state_dict' in checkpoint:
            model.encoder.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.encoder.load_state_dict(checkpoint)
        print(f"成功加载权重: {checkpoint_path}")
    except RuntimeError as e:
        print(f"[警告] 权重加载失败，原因: {e}\n将使用新结构的随机初始化权重进行评估/可视化。")
else:
    print(f"[警告] 未找到权重文件: {checkpoint_path}，将使用随机初始化权重。")
model.eval()

# 7. 推理与保存
with torch.no_grad():
    recon_img = model(masked_img_tensor).cpu().numpy()[0, 0]

compare_dir = project_root / "data" / "ssl_compare"
compare_dir.mkdir(parents=True, exist_ok=True)
plt.imsave(compare_dir / "inspect_masked.png", masked_img[0], cmap="gray")
plt.imsave(compare_dir / "inspect_mask.png", loss_mask, cmap="gray")
plt.imsave(compare_dir / "inspect_recon.png", recon_img, cmap="gray")
plt.imsave(compare_dir / "inspect_original.png", original_img[0], cmap="gray")
print(f"[对比模式] inspect已保存Masked Input、Mask Area、Reconstructed、Original到: {compare_dir}") 