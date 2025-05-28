import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import numpy as np
import json
import os # os.cpu_count()

# --- 配置导入 ---
# (与json_generator.py和sem_segmentation_dataset.py类似的导入逻辑)
try:
    from configs import base as cfg_base
    from configs import json_config as cfg_json_gen
    from datasets.sem_datasets import SemSegmentationDataset # 您的Dataset类
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
            # Dataset类在datasets子目录下
            datasets_module_path = project_root_for_config / "datasets"
            if str(datasets_module_path) not in sys.path:
                 sys.path.insert(0, str(datasets_module_path))
            from datasets.sem_datasets import SemSegmentationDataset

        except ImportError as e_inner:
            print(f"Error: Could not import required modules: {e_inner}")
            print(f"Attempted project root for config/datasets: {project_root_for_config}")
            sys.exit(1)
    else:
        raise

def calculate_mean_std(dataset: torch.utils.data.Dataset,
                       batch_size: int = 32,
                       num_workers: int = 0) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    if len(dataset) == 0:
        print("Dataset is empty, cannot calculate mean/std.")
        return None, None

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    channels = -1
    for img_tensor_sample, _ in loader: # Try to get first valid batch
        if isinstance(img_tensor_sample, torch.Tensor) and img_tensor_sample.numel() > 0:
            channels = img_tensor_sample.shape[1] # (B, C, H, W)
            break
    if channels == -1:
        print("Error: Could not determine channel count from dataset. Dataset might be empty or returning invalid samples.")
        return None, None

    channel_sum = torch.zeros(channels)
    channel_sum_sq = torch.zeros(channels)
    num_pixels_total_per_channel = 0 # Accumulate total number of pixels (H*W) over all images
    total_images_processed = 0

    print(f"Calculating mean and std over {len(dataset)} samples using {channels} channel(s)...")
    for i, (images_batch, _) in enumerate(loader): # Dataset returns (image, mask)
        if not isinstance(images_batch, torch.Tensor) or images_batch.numel() == 0:
            print(f"Warning: Skipping empty or invalid batch {i}")
            continue

        if images_batch.max() > 1.01 or images_batch.min() < -0.01: # 允许微小误差
            print(f"Warning: Batch {i} image values seem not to be in [0,1] range "
                  f"(min: {images_batch.min():.2f}, max: {images_batch.max():.2f}). "
                  "Mean/std calculation might be incorrect if ToTensor() scaling was missed or done differently in Dataset.")

        channel_sum += torch.sum(images_batch, dim=[0, 2, 3])
        channel_sum_sq += torch.sum(images_batch ** 2, dim=[0, 2, 3])

        num_pixels_in_batch_per_channel = images_batch.size(0) * images_batch.size(2) * images_batch.size(3)
        num_pixels_total_per_channel += num_pixels_in_batch_per_channel
        total_images_processed += images_batch.size(0)

        if (i + 1) % (max(1, len(loader) // 10)) == 0: # Print progress roughly 10 times
            print(f"  Processed batch {i+1}/{len(loader)} ({total_images_processed} images)...")

    if total_images_processed == 0 or num_pixels_total_per_channel == 0:
        print("No valid samples or pixels processed, cannot calculate mean/std.")
        return None, None

    mean = channel_sum / num_pixels_total_per_channel
    # Variance = E[X^2] - (E[X])^2
    variance = (channel_sum_sq / num_pixels_total_per_channel) - (mean ** 2)
    std = torch.sqrt(variance)

    std[torch.isnan(std)] = 0
    std[std < 1e-6] = 1e-6 # 避免标准差过小导致后续除以零的问题，设定一个小的下限

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

    potential_stats_json_path = json_dir_abs / train_json_filename_for_stats
    if not potential_stats_json_path.exists():
        print(f"Specified stats JSON '{train_json_filename_for_stats}' not found.")
        master_labeled_path = json_dir_abs / "master_labeled_dataset.json"
        if master_labeled_path.exists():
            print(f"Attempting to use '{master_labeled_path.name}' instead for statistics.")
            stats_json_path = master_labeled_path
            train_json_filename_for_stats = master_labeled_path.name # 更新用于日志的文件名
        else:
            print(f"Error: Neither '{train_json_filename_for_stats}' nor 'master_labeled_dataset.json' "
                  f"found in '{json_dir_abs}'.")
            print("Please run json_generator.py first to create the dataset JSON file(s).")
            sys.exit(1)
    else:
        stats_json_path = potential_stats_json_path


    print(f"Using JSON file for statistics: {stats_json_path}")

    try:
        stats_calc_dataset = SemSegmentationDataset(
            json_file_identifier=str(train_json_filename_for_stats), # 传递文件名
            project_root=project_root,
            input_depth_from_config=cfg_json_gen.INPUT_DEPTH,
            class_mapping_from_config=cfg_base.MAPPING,
            json_dir_name_relative_to_project=cfg_json_gen.JSON_OUTPUT_DIR_NAME,
            image_transform=None, # Dataset内部会做基本的ToTensor和[0,1]缩放
            augmentations=None    # 不进行随机增强
        )
    except FileNotFoundError as e:
        print(f"Error during Dataset instantiation for statistics: {e}")
        print("Ensure the JSON file and the image/mask paths it references are correct.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during Dataset instantiation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


    if len(stats_calc_dataset) > 0:
        # 确定num_workers，可以稍微保守一些以避免内存问题
        available_cpus = os.cpu_count()
        num_workers = min(available_cpus // 2 if available_cpus else 2, 4)
        print(f"Using {num_workers} workers for DataLoader.")

        calculated_mean, calculated_std = calculate_mean_std(
            stats_calc_dataset,
            batch_size=getattr(cfg_json_gen, 'STATS_CALC_BATCH_SIZE', 16), # 可在json_config中配置
            num_workers=num_workers
        )

        if calculated_mean is not None and calculated_std is not None:
            # 将统计量保存到文件，供 augmentations_utils.py 读取
            stats_output_file = json_dir_abs / "dataset_stats.json" # 固定文件名
            stats_data_to_save = {
                "mean": calculated_mean,
                "std": calculated_std,
                "source_json": str(train_json_filename_for_stats), # 记录来源
                "input_depth_at_calculation": cfg_json_gen.INPUT_DEPTH # 记录计算时的深度
            }
            with open(stats_output_file, 'w') as f:
                json.dump(stats_data_to_save, f, indent=4)
            print(f"\nDataset statistics (mean, std) saved to: {stats_output_file}")
            print("You can now use these values for normalization in your training pipeline.")
            print("`utils/augmentations_utils.py` will attempt to load this file automatically.")
            print("\n--- Example for configs/base.py (if you want to hardcode them as well) ---")
            print(f"IMG_MEAN = {calculated_mean}")
            print(f"IMG_STD = {calculated_std}")
            print("---------------------------------------------------------------------------")
        else:
            print("Failed to calculate dataset statistics.")
    else:
        print(f"The dataset loaded from '{train_json_filename_for_stats}' is empty. Cannot calculate statistics.")