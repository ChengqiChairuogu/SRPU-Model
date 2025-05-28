import os
import json
import re
from collections import defaultdict
from pathlib import Path
import argparse
import random
import sys
from typing import Optional, List, Tuple, Dict, Any # 确保导入了必要的类型

# --- 全局变量，将在 main 函数中根据配置设置 ---
RAW_IMAGE_SOURCE_DIR_ABSOLUTE_FROM_CONFIG: Optional[Path] = None
# LABELED_MASK_DIR_ABSOLUTE_FROM_CONFIG: Optional[Path] = None # 不再需要这个全局变量直接被get_mask_path使用
JSON_OUTPUT_DIR_ABSOLUTE_FROM_CONFIG: Optional[Path] = None

# --- 配置导入 ---
try:
    from configs import json_config as cfg_json
    from configs import base as cfg_base
except ImportError:
    if __name__ == '__main__' and __package__ is None:
        current_script_path = Path(__file__).resolve()
        project_root_attempt1 = current_script_path.parent.parent 
        project_root_attempt2 = current_script_path.parent 

        if (project_root_attempt1 / "configs").is_dir():
            project_root_for_config = project_root_attempt1
        elif (project_root_attempt2 / "configs").is_dir():
            project_root_for_config = project_root_attempt2
        else:
            print(f"Error: Could not automatically determine project root containing 'configs' directory.")
            print(f"Attempted paths: {project_root_attempt1}, {project_root_attempt2}")
            print(f"Please ensure the script is run from the project root or 'configs' is in PYTHONPATH.")
            sys.exit(1)

        sys.path.insert(0, str(project_root_for_config))
        try:
            from configs import json_config as cfg_json
            from configs import base as cfg_base
        except ImportError as e:
            print(f"Error: Could not import configuration files even after attempting to set project root.")
            print(f"Attempted project root for config: {project_root_for_config}")
            print(f"Original ImportError: {e}")
            sys.exit(1)
    else: 
        raise


def get_mask_path_for_raw_image(
    raw_image_path: Path, # 原始图像的绝对路径
    labeled_mask_dir_abs: Path, # <--- 明确接收此参数
    mask_extensions: List[str]
) -> Optional[Path]:
    """
    根据原始图像路径推断其对应的掩码路径。
    核心逻辑：掩码文件名主体与原始图像文件名主体相同。
    """
    if RAW_IMAGE_SOURCE_DIR_ABSOLUTE_FROM_CONFIG is None: # 这个全局变量仍然用于计算相对路径
        raise ValueError("Global config path RAW_IMAGE_SOURCE_DIR_ABSOLUTE_FROM_CONFIG has not been set in main().")

    try:
        # 获取原始图像相对于其配置的源目录的相对路径
        relative_to_raw_source = raw_image_path.relative_to(RAW_IMAGE_SOURCE_DIR_ABSOLUTE_FROM_CONFIG)
    except ValueError:
        relative_to_raw_source = Path(raw_image_path.name) 

    for ext in mask_extensions:
        # 构建掩码在传入的 labeled_mask_dir_abs 下的潜在相对路径，然后得到绝对路径
        potential_mask_relative_path = relative_to_raw_source.with_name(f"{raw_image_path.stem}{ext}")
        potential_mask_absolute_path = labeled_mask_dir_abs / potential_mask_relative_path # <--- 使用传入的参数
        
        if potential_mask_absolute_path.exists():
            return potential_mask_absolute_path
            
    return None


def build_sequences(
    image_info_list: List[Tuple[str, int, Path, Optional[Path]]], 
    depth: int,
    raw_image_source_dir_abs: Path, 
    labeled_mask_dir_abs: Optional[Path]
) -> List[Dict[str, Any]]:
    """
    从解析后的图像信息列表构建3D序列样本。
    路径将是相对于各自在JSON中记录的根目录的相对路径。
    """
    sequences: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for seq_key, frame_num, abs_img_path, abs_mask_path in image_info_list:
        sequences[seq_key].append({
            "frame_num": frame_num,
            "image_file_abs": abs_img_path,
            "mask_file_abs": abs_mask_path
        })

    output_samples: List[Dict[str, Any]] = []
    half_depth = depth // 2 

    for seq_key, frames_in_seq in sequences.items():
        frames_in_seq.sort(key=lambda x: x["frame_num"]) 
        num_frames_in_sequence = len(frames_in_seq)

        if num_frames_in_sequence == 0:
            continue

        for i in range(num_frames_in_sequence): 
            current_frame_info = frames_in_seq[i]
            input_frame_paths_relative_to_raw_source: List[str] = []
            
            for offset in range(-half_depth, half_depth + 1):
                target_idx_in_sequence = i + offset
                
                if target_idx_in_sequence < 0:
                    frame_to_append_abs = frames_in_seq[0]["image_file_abs"]
                elif target_idx_in_sequence >= num_frames_in_sequence:
                    frame_to_append_abs = frames_in_seq[-1]["image_file_abs"]
                else:
                    frame_to_append_abs = frames_in_seq[target_idx_in_sequence]["image_file_abs"]
                
                try:
                    relative_path = frame_to_append_abs.relative_to(raw_image_source_dir_abs)
                    input_frame_paths_relative_to_raw_source.append(str(relative_path))
                except ValueError: 
                    input_frame_paths_relative_to_raw_source.append(str(frame_to_append_abs))

            if len(input_frame_paths_relative_to_raw_source) != depth:
                continue 

            mask_file_relative_to_labeled_dir: Optional[str] = None
            if current_frame_info["mask_file_abs"] and labeled_mask_dir_abs:
                try:
                    mask_file_relative_to_labeled_dir = str(
                        current_frame_info["mask_file_abs"].relative_to(labeled_mask_dir_abs)
                    )
                except ValueError:
                    mask_file_relative_to_labeled_dir = str(current_frame_info["mask_file_abs"])
            
            center_frame_relative_path_str = input_frame_paths_relative_to_raw_source[half_depth]
            center_frame_stem = Path(center_frame_relative_path_str).stem 

            sample = {
                "sample_id": f"{seq_key}_{center_frame_stem}_centerFrameIdx{current_frame_info['frame_num']:04d}",
                "frames": input_frame_paths_relative_to_raw_source, 
                "mask_file": mask_file_relative_to_labeled_dir  
            }
            output_samples.append(sample)
            
    return output_samples


def scan_all_images_and_match_masks(
    raw_image_source_dir_abs: Path,
    labeled_mask_dir_abs: Path, # 这个参数会被传递给 get_mask_path_for_raw_image
    filename_pattern: re.Pattern,
    image_extensions: List[str],
    mask_extensions: List[str]
) -> Tuple[List[Tuple[str, int, Path, Optional[Path]]], List[Tuple[str, int, Path, Optional[Path]]]]:
    """扫描原始图像，匹配掩码，返回已标注和未标注图像的信息列表。"""
    all_raw_images_info: List[Tuple[str, int, Path, Optional[Path]]] = []

    print(f"Scanning raw images in: {raw_image_source_dir_abs}")
    if not raw_image_source_dir_abs.exists():
        print(f"Error: Raw image source directory not found: {raw_image_source_dir_abs}")
        return [], []
    if not labeled_mask_dir_abs.exists(): 
        print(f"Warning: Labeled mask directory '{labeled_mask_dir_abs}' not found. All images will be treated as unlabeled.")

    for abs_image_path in raw_image_source_dir_abs.rglob('*'):
        if abs_image_path.is_file() and abs_image_path.suffix.lower() in image_extensions:
            match = filename_pattern.match(abs_image_path.name)
            sequence_key = "default_seq" 
            frame_number = 0

            if match:
                sequence_key_from_name = match.group(1)
                frame_number_str = match.group(2)
                try: frame_number = int(frame_number_str)
                except ValueError: frame_number = 0
                
                try:
                    parent_dir_relative = abs_image_path.parent.relative_to(raw_image_source_dir_abs)
                    if str(parent_dir_relative) != '.': 
                        sequence_key = f"{str(parent_dir_relative).replace(os.sep, '_')}_{sequence_key_from_name}" if sequence_key_from_name else str(parent_dir_relative).replace(os.sep, '_')
                    else: 
                        sequence_key = sequence_key_from_name if sequence_key_from_name else abs_image_path.stem
                except ValueError: 
                    sequence_key = sequence_key_from_name if sequence_key_from_name else abs_image_path.stem
            else: 
                try:
                    parent_dir_relative = abs_image_path.parent.relative_to(raw_image_source_dir_abs)
                    sequence_key = str(parent_dir_relative).replace(os.sep, '_') if str(parent_dir_relative) != '.' else abs_image_path.stem
                except ValueError:
                     sequence_key = abs_image_path.stem

            abs_mask_path = None
            if labeled_mask_dir_abs.exists(): 
                 # 调用时传递 labeled_mask_dir_abs
                 abs_mask_path = get_mask_path_for_raw_image(abs_image_path, labeled_mask_dir_abs, mask_extensions)
            all_raw_images_info.append((sequence_key, frame_number, abs_image_path, abs_mask_path))
    
    labeled_info = [info for info in all_raw_images_info if info[3] is not None]
    unlabeled_info = [info for info in all_raw_images_info if info[3] is None]
    return labeled_info, unlabeled_info

def save_json_dataset(
    samples: List[Dict[str, Any]],
    dataset_name: str,
    description: str,
    input_depth: int,
    output_json_path: Path,
    is_labeled_dataset: bool 
):
    """保存数据集信息为JSON文件。路径名从config中获取，用于JSON记录。"""
    json_data: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "description": description,
        "root_raw_image_dir": cfg_json.RAW_IMAGE_SOURCE_DIR_NAME, 
        "input_depth": input_depth,
        "num_samples": len(samples),
        "samples": samples
    }
    if is_labeled_dataset and hasattr(cfg_json, 'LABELED_MASK_DIR_NAME') and cfg_json.LABELED_MASK_DIR_NAME:
        json_data["root_labeled_mask_dir"] = cfg_json.LABELED_MASK_DIR_NAME

    with open(output_json_path, 'w') as f:
        json.dump(json_data, f, indent=4)
    print(f"Dataset JSON saved to: {output_json_path}")


def main(args: argparse.Namespace):
    global RAW_IMAGE_SOURCE_DIR_ABSOLUTE_FROM_CONFIG 
    # global LABELED_MASK_DIR_ABSOLUTE_FROM_CONFIG # No longer needed as global for get_mask_path
    global JSON_OUTPUT_DIR_ABSOLUTE_FROM_CONFIG


    project_root_abs = cfg_base.PROJECT_ROOT.resolve()
    RAW_IMAGE_SOURCE_DIR_ABSOLUTE_FROM_CONFIG = (project_root_abs / cfg_json.RAW_IMAGE_SOURCE_DIR_NAME).resolve()
    # labeled_mask_dir_abs is now a local variable in main, passed as arg
    labeled_mask_dir_abs_local = (project_root_abs / cfg_json.LABELED_MASK_DIR_NAME).resolve()
    JSON_OUTPUT_DIR_ABSOLUTE_FROM_CONFIG = (project_root_abs / cfg_json.JSON_OUTPUT_DIR_NAME).resolve()
    
    input_depth = args.depth if args.depth is not None else cfg_json.INPUT_DEPTH
    filename_pattern_str = args.pattern if args.pattern else cfg_json.FILENAME_PATTERN_STR
    image_extensions = args.img_ext if args.img_ext else cfg_json.EXPECTED_IMAGE_EXTENSIONS
    mask_extensions = args.mask_ext if args.mask_ext else cfg_json.EXPECTED_MASK_EXTENSIONS
    
    val_split_ratio = args.val_split if args.val_split is not None else cfg_json.DEFAULT_VAL_SPLIT
    test_split_ratio = args.test_split if args.test_split is not None else cfg_json.DEFAULT_TEST_SPLIT
    random_seed = args.seed if args.seed is not None else cfg_json.DEFAULT_RANDOM_SEED

    filename_pattern_compiled = re.compile(filename_pattern_str, re.IGNORECASE)
    JSON_OUTPUT_DIR_ABSOLUTE_FROM_CONFIG.mkdir(parents=True, exist_ok=True)

    if args.mode == "generate_all" or args.mode == "generate_labeled_unlabeled":
        print(f"Mode: Generating labeled and unlabeled datasets using paths from configuration files...")
        labeled_info, unlabeled_info = scan_all_images_and_match_masks(
            RAW_IMAGE_SOURCE_DIR_ABSOLUTE_FROM_CONFIG,
            labeled_mask_dir_abs_local, # Pass the local absolute path
            filename_pattern_compiled,
            image_extensions,
            mask_extensions
        )
        print(f"Found {len(labeled_info)} raw image frames with corresponding masks.")
        print(f"Found {len(unlabeled_info)} raw image frames without corresponding masks.")

        labeled_dataset_samples = build_sequences(
            labeled_info, 
            input_depth, 
            RAW_IMAGE_SOURCE_DIR_ABSOLUTE_FROM_CONFIG, 
            labeled_mask_dir_abs_local # Pass the local absolute path
        )
        master_labeled_json_path = JSON_OUTPUT_DIR_ABSOLUTE_FROM_CONFIG / "master_labeled_dataset.json"
        save_json_dataset(
            labeled_dataset_samples, 
            "SEM_Labeled_Dataset_Master", 
            "Master list of all labeled SEM images.",
            input_depth, 
            master_labeled_json_path,
            is_labeled_dataset=True 
        )

        unlabeled_dataset_samples = build_sequences(
            unlabeled_info, 
            input_depth, 
            RAW_IMAGE_SOURCE_DIR_ABSOLUTE_FROM_CONFIG, 
            None 
        )
        save_json_dataset(
            unlabeled_dataset_samples, 
            "SEM_Unlabeled_Dataset_Master", 
            "Master list of all unlabeled SEM images.",
            input_depth, 
            JSON_OUTPUT_DIR_ABSOLUTE_FROM_CONFIG / "master_unlabeled_dataset.json",
            is_labeled_dataset=False 
        )
        
        if args.mode == "generate_all":
            if labeled_dataset_samples: 
                args.mode = "split_labeled" 
                args.input_json = master_labeled_json_path.name 
                print(f"\nContinuing to split '{master_labeled_json_path.name}'...")
            else:
                 print("\nNo labeled data found to split. Skipping split_labeled mode for 'generate_all'.")
                 return 

    if args.mode == "split_labeled":
        if not args.input_json:
            print("Error: --input_json filename is required for 'split_labeled' mode.")
            return
        
        input_json_path_to_split = JSON_OUTPUT_DIR_ABSOLUTE_FROM_CONFIG / args.input_json
        if not input_json_path_to_split.exists():
            print(f"Error: Input JSON file for splitting not found: {input_json_path_to_split}")
            return

        print(f"Mode: Splitting labeled dataset from {input_json_path_to_split}...")
        try:
            with open(input_json_path_to_split, 'r') as f:
                source_labeled_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {input_json_path_to_split}: {e}")
            return
        except Exception as e:
            print(f"Error reading {input_json_path_to_split}: {e}")
            return
            
        all_labeled_samples = source_labeled_data.get("samples", [])
        if not all_labeled_samples:
            print(f"No samples found in '{args.input_json}' to split.")
            return
        
        source_dataset_name = source_labeled_data.get("dataset_name", "Unknown_Labeled_Dataset")
        source_input_depth = source_labeled_data.get("input_depth", input_depth) 

        random.seed(random_seed)
        shuffled_samples = random.sample(all_labeled_samples, len(all_labeled_samples))

        num_samples = len(shuffled_samples)
        num_val = int(num_samples * val_split_ratio)
        num_test = int(num_samples * test_split_ratio)
        num_train = num_samples - num_val - num_test

        if num_train <= 0 :
            print(f"Error: Not enough samples ({num_samples}) to create a non-empty training set with val_split={val_split_ratio} and test_split={test_split_ratio}.")
            return

        train_samples = shuffled_samples[:num_train]
        val_samples = shuffled_samples[num_train : num_train + num_val]
        test_samples = shuffled_samples[num_train + num_val:]

        print(f"Total labeled samples from '{args.input_json}': {num_samples}")
        print(f"  Training samples: {len(train_samples)}")
        print(f"  Validation samples: {len(val_samples)}")
        print(f"  Test samples: {len(test_samples)}")

        output_stem = Path(args.input_json).stem 
        save_json_dataset(
            train_samples, f"{output_stem}_Train", "Training split.",
            source_input_depth, JSON_OUTPUT_DIR_ABSOLUTE_FROM_CONFIG / f"{output_stem}_train.json", True
        )
        if val_samples:
            save_json_dataset(
                val_samples, f"{output_stem}_Validation", "Validation split.",
                source_input_depth, JSON_OUTPUT_DIR_ABSOLUTE_FROM_CONFIG / f"{output_stem}_val.json", True
            )
        if test_samples:
            save_json_dataset(
                test_samples, f"{output_stem}_Test", "Test split.",
                source_input_depth, JSON_OUTPUT_DIR_ABSOLUTE_FROM_CONFIG / f"{output_stem}_test.json", True
            )
    
    elif args.mode not in ["generate_all", "generate_labeled_unlabeled"]: 
        print(f"Unknown or unprocessed mode: {args.mode}. Please choose from 'generate_all', 'generate_labeled_unlabeled', 'split_labeled'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate and manage JSON dataset files for SEM image segmentation.",
        formatter_class=argparse.RawTextHelpFormatter 
    )
    parser.add_argument(
        "--mode", type=str, default="generate_all",
        choices=["generate_all", "generate_labeled_unlabeled", "split_labeled"],
        help="Operation mode:\n"
             "'generate_all': Scan raw/labeled, create master JSONs, then split labeled into train/val/test.\n"
             "'generate_labeled_unlabeled': Scan raw/labeled and create master_labeled_dataset.json and master_unlabeled_dataset.json.\n"
             "'split_labeled': Split an existing labeled JSON into train/val/test sets."
    )
    parser.add_argument(
        "--input_json", type=str,
        help=("Filename of an existing labeled JSON (it will be looked for in the directory "
              "specified by JSON_OUTPUT_DIR_NAME in configs/json_config.py). "
              "Required for 'split_labeled' mode.")
    )
    parser.add_argument("--val_split", type=float, default=None, help="Validation split ratio (overrides config, e.g., 0.15).")
    parser.add_argument("--test_split", type=float, default=None, help="Test split ratio (overrides config, e.g., 0.15).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for splitting (overrides config).")
    
    parser.add_argument("--depth", type=int, default=None, help="Input depth for 3D sequences (overrides config from configs.json_config).")
    parser.add_argument("--pattern", type=str, default=None, help="Regex pattern for filenames (overrides config from configs.json_config).")
    parser.add_argument("--img_ext", nargs='+', default=None, help="Image extensions (overrides config, e.g., .tif .png).")
    parser.add_argument("--mask_ext", nargs='+', default=None, help="Mask extensions (overrides config, e.g., .png).")
    
    args = parser.parse_args()
    main(args)