import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import sys
from typing import Tuple, List

# --- 配置导入 ---
try:
    from configs import base as cfg_base
    from configs import json_config as cfg_json_gen
    from configs.train import train_config as cfg_train
except ImportError:
    if __name__ == '__main__' and __package__ is None:
        current_script_path = Path(__file__).resolve()
        project_root_for_import = current_script_path.parent.parent
        if str(project_root_for_import) not in sys.path:
            sys.path.insert(0, str(project_root_for_import))
        from configs import base as cfg_base
        from configs import json_config as cfg_json_gen
        from configs.train import train_config as cfg_train
    else:
        raise

# --- 为统计计算专门定义的简单数据集 ---
class StatsCalculatorDataset(Dataset):
    """
    一个更简单、更正确的数据集，用于加载单张原始图像以计算统计数据。
    它会创建一个包含数据集中所有独立图像帧路径的列表。
    """
    def __init__(self, json_file_path: Path, project_root: Path):
        self.project_root = project_root.resolve()
        
        if not json_file_path.exists():
            raise FileNotFoundError(f"数据集JSON文件未找到: {json_file_path}")

        with open(json_file_path, 'r') as f:
            data_info = json.load(f)

        self.raw_image_root = self.project_root / data_info["root_raw_image_dir"]
        
        # **核心修改**：创建一个包含所有独立帧路径的扁平列表
        self.all_frames_paths = []
        for sample in data_info.get("samples", []):
            for frame_filename in sample.get("frames", []):
                self.all_frames_paths.append(self.raw_image_root / frame_filename)
        
        # 去重，以防同一个文件在JSON中被多次引用
        self.all_frames_paths = sorted(list(set(self.all_frames_paths)))
        print(f"共找到 {len(self.all_frames_paths)} 张独立的图像文件用于统计。")


    def __len__(self) -> int:
        # 长度是所有独立帧的总数
        return len(self.all_frames_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # 每次只加载和处理一张图片
        image_path = self.all_frames_paths[idx]
        
        with Image.open(image_path) as img:
            img = img.convert('L')
            # 统一尺寸，这仍然是必要的，因为原始文件分辨率可能不同
            target_size = (cfg_base.IMAGE_WIDTH, cfg_base.IMAGE_HEIGHT)
            img = img.resize(target_size, Image.Resampling.BILINEAR)
            
            img_np = np.array(img, dtype=np.float32) / 255.0
        
        # 返回单张图片的张量，形状为 (1, H, W)
        return torch.from_numpy(img_np).unsqueeze(0)


def calculate_mean_std(dataloader: DataLoader) -> Tuple[List[float], List[float]]:
    """
    遍历数据加载器以计算均值和标准差。
    现在它处理的是单通道图像。
    """
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    
    for images in tqdm(dataloader, desc="Calculating Stats"):
        # images 的形状是 (B, 1, H, W)
        channels_sum += torch.mean(images) # 直接对整个批次求均值
        channels_squared_sum += torch.mean(images**2)
        num_batches += 1
        
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean**2)**0.5
    
    # 因为是单通道灰度图，所以均值和标准差都只有一个值
    # 但我们的配置和归一化步骤期望一个列表，所以我们用这个值来构建一个列表
    final_mean = [mean.item()] * cfg_base.INPUT_DEPTH
    final_std = [std.item()] * cfg_base.INPUT_DEPTH
    
    return final_mean, final_std

# --- 主执行函数 ---
def main():
    print("--- 正在计算数据集的均值和标准差 (单图像模式) ---")
    project_root = cfg_base.PROJECT_ROOT.resolve()
    json_dir = project_root / cfg_json_gen.JSON_OUTPUT_DIR_NAME
    json_for_stats_path = json_dir / cfg_train.TRAIN_JSON_NAME
    print(f"使用JSON文件进行统计: {json_for_stats_path}")

    try:
        stats_calc_dataset = StatsCalculatorDataset(
            json_file_path=json_for_stats_path,
            project_root=project_root
        )
    except Exception as e:
        print(f"数据集实例化过程中发生意外错误: {e}")
        return

    dataloader = DataLoader(stats_calc_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    mean, std = calculate_mean_std(dataloader)

    print("\n--- 计算完成 ---")
    print(f"计算出的单通道均值 (Mean): {mean[0]}")
    print(f"计算出的单通道标准差 (Std Dev): {std[0]}")
    print(f"扩展到 {cfg_base.INPUT_DEPTH} 个通道后的均值: {mean}")
    print(f"扩展到 {cfg_base.INPUT_DEPTH} 个通道后的标准差: {std}")

    stats_dict = {
        "mean": mean,
        "std": std,
        "input_depth_at_calculation": cfg_base.INPUT_DEPTH,
        "source_json": str(json_for_stats_path.name)
    }
    output_stats_file = json_dir / "dataset_stats.json"
    with open(output_stats_file, 'w') as f:
        json.dump(stats_dict, f, indent=4)
    print(f"\n统计数据已保存至: {output_stats_file}")

if __name__ == '__main__':
    main()