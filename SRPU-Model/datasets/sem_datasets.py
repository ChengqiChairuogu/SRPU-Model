import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import numpy as np
import json
import os
from typing import Tuple, Optional, List # <--- 新增导入 Tuple

# --- 配置导入 ---
try:
    from configs import base as cfg_base
    from configs import json_config as cfg_json_gen
    from datasets.sem_datasets import SemSegmentationDataset
except ImportError:
    if __name__ == '__main__' and __package__ is None:
        current_script_path = Path(__file__).resolve()
        project_root_for_config = current_script_path.parent.parent
        if not (project_root_for_config / "configs").is_dir():
            project_root_for_config = current_script_path.parent
            if not (project_root_for_config / "configs").is_dir() and \
               (project_root_for_config.parent / "configs").exists():
                 project_root_for_config = project_root_for_config.parent
        
        if project_root_for_config and str(project_root_for_config) not in sys.path:
            sys.path.insert(0, str(project_root_for_config))
        try:
            from configs import base as cfg_base
            from configs import json_config as cfg_json_gen
            from datasets.sem_datasets import SemSegmentationDataset
        except ImportError as e_inner:
            print(f"Error: Could not import required modules: {e_inner}")
            sys.exit(1)
    else:
        raise

def calculate_mean_std(dataset: torch.utils.data.Dataset,
                       batch_size: int = 32,
                       num_workers: int = 0) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """
    Calculates mean and std for a given dataset.
    Assumes dataset's __getitem__ returns (image_tensor, mask_tensor),
    and image_tensor is already scaled to [0, 1].
    """
    if len(dataset) == 0:
        print("Dataset is empty, cannot calculate mean/std.")
        return None, None

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    channels = -1
    try:
        first_img_tensor, _ = dataset[0]
        if isinstance(first_img_tensor, torch.Tensor) and first_img_tensor.numel() > 0:
            channels = first_img_tensor.shape[0]
        else:
            for img_batch_sample, _ in loader:
                if isinstance(img_batch_sample, torch.Tensor) and img_batch_sample.numel() > 0:
                    channels = img_batch_sample.shape[1]
                    break
    except Exception as e:
        print(f"Error getting first sample to determine channel count: {e}")
        return None, None

    if channels == -1 or channels == 0:
        print(f"Error: Could not determine channel count ({channels}) from dataset.")
        return None, None

    channel_sum = torch.zeros(channels)
    channel_sum_sq = torch.zeros(channels)
    num_pixels_total_for_mean_std = 0
    total_images_processed = 0

    print(f"Calculating mean and std over {len(dataset)} samples using {channels} channel(s)...")
    for i, (images_batch, _) in enumerate(loader):
        if not isinstance(images_batch, torch.Tensor) or images_batch.numel() == 0:
            print(f"Warning: Skipping empty or invalid batch {i}")
            continue
        
        if images_batch.max() > 1.01 or images_batch.min() < -0.01:
            print(f"Warning: Batch {i} image values seem not to be in [0,1] range "
                  f"(min: {images_batch.min():.2f}, max: {images_batch.max():.2f}).")

        channel_sum += torch.sum(images_batch, dim=[0, 2, 3])
        channel_sum_sq += torch.sum(images_batch ** 2, dim=[0, 2, 3])
        
        num_pixels_total_for_mean_std += images_batch.size(0) * images_batch.size(2) * images_batch.size(3)
        total_images_processed += images_batch.size(0)

        if (i + 1) % (max(1, len(loader) // 10)) == 0:
            print(f"  Processed batch {i+1}/{len(loader)} ({total_images_processed} images)...")
            
    if total_images_processed == 0 or num_pixels_total_for_mean_std == 0:
        print("No valid samples or pixels processed, cannot calculate mean/std.")
        return None, None

    mean = channel_sum / num_pixels_total_for_mean_std
    variance = (channel_sum_sq / num_pixels_total_for_mean_std) - (mean ** 2)
    std = torch.sqrt(torch.abs(variance))
    std[std < 1e-6] = 1e-6

    mean_list = mean.tolist()
    std_list = std.tolist()
    print(f"\nCalculation complete.")
    print(f"Processed {total_images_processed} images in total.")
    print(f"Calculated mean: {mean_list}")
    print(f"Calculated std: {std_list}")
    return mean_list, std_list

if __name__ == '__main__':
    print("--- Calculating Dataset Mean and Standard Deviation ---")
    
    project_root = cfg_base.PROJECT_ROOT.resolve()
    json_dir_abs = project_root / cfg_json_gen.JSON_OUTPUT_DIR_NAME
    
    train_json_filename_for_stats = getattr(cfg_json_gen, 'STATS_CALCULATION_JSON', "master_labeled_dataset_train.json")
    
    stats_json_path = json_dir_abs / train_json_filename_for_stats

    if not stats_json_path.exists():
        print(f"Specified stats JSON '{train_json_filename_for_stats}' not found.")
        master_labeled_path = json_dir_abs / "master_labeled_dataset.json"
        if master_labeled_path.exists():
            print(f"Attempting to use '{master_labeled_path.name}' instead for statistics.")
            stats_json_path = master_labeled_path
            train_json_filename_for_stats = master_labeled_path.name
        else:
            print(f"Error: Neither '{train_json_filename_for_stats}' nor 'master_labeled_dataset.json' "
                  f"found in '{json_dir_abs}'.")
            print("Please run json_generator.py first to create it.")
            sys.exit(1)
    
    print(f"Using JSON file for statistics: {stats_json_path}")

    try:
        # --- 修正部分 ---
        # 移除了 'image_transform' 这个不再被接受的参数
        # 并且添加了 patch_size 和 stride，即使在统计时不直接使用，
        # 但 __init__ 方法现在需要它们。我们可以从配置中获取。
        stats_calc_dataset = SemSegmentationDataset(
            json_file_identifier=str(train_json_filename_for_stats),
            project_root=project_root,
            input_depth_from_config=cfg_json_gen.INPUT_DEPTH,
            class_mapping_from_config=cfg_base.MAPPING,
            json_dir_name_relative_to_project=cfg_json_gen.JSON_OUTPUT_DIR_NAME,
            patch_size=cfg_base.PATCH_SIZE, # 从 base config 获取
            stride=cfg_base.STRIDE,       # 从 base config 获取
            augmentations=None # 确保没有增强，使用Dataset的默认ToTensor和缩放
        )
        # --- 修正结束 ---
    except Exception as e:
        print(f"An unexpected error occurred during Dataset instantiation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    if len(stats_calc_dataset) > 0:
        available_cpus = os.cpu_count()
        num_workers = min(available_cpus // 2 if available_cpus else 2, 4)
        print(f"Using {num_workers} workers for DataLoader.")

        calculated_mean, calculated_std = calculate_mean_std(
            stats_calc_dataset,
            batch_size=getattr(cfg_json_gen, 'STATS_CALC_BATCH_SIZE', 16),
            num_workers=num_workers
        )
        
        if calculated_mean is not None and calculated_std is not None:
            stats_output_file = json_dir_abs / "dataset_stats.json"
            stats_data_to_save = {
                "mean": calculated_mean,
                "std": calculated_std,
                "source_json": str(train_json_filename_for_stats),
                "input_depth_at_calculation": cfg_json_gen.INPUT_DEPTH
            }
            with open(stats_output_file, 'w') as f:
                json.dump(stats_data_to_save, f, indent=4)
            print(f"\nDataset statistics (mean, std) saved to: {stats_output_file}")
            print("You can now use these values for normalization in your training pipeline.")
        else:
            print("Failed to calculate dataset statistics.")
    else:
        print(f"The dataset loaded from '{train_json_filename_for_stats}' is empty. Cannot calculate statistics.")
