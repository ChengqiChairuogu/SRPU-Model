import sys
from pathlib import Path
import json
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
from typing import Union, Tuple, Dict, Any

# --- 动态添加项目根目录到Python路径 ---
try:
    # **关键修正**: 使用 .parent.parent 来获取正确的项目根目录
    project_root = Path(__file__).resolve().parent.parent
except NameError:
    project_root = Path.cwd()

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from configs import json_config as cfg_json_gen
    from configs.train import train_config as cfg_train
    from configs.finetune import finetune_config as cfg_finetune
except ImportError as e:
    print(f"错误：无法导入配置文件。请确保此脚本位于项目根目录下，并且 'configs' 文件夹存在。")
    print(f"具体错误: {e}")
    sys.exit(1)


def get_image_size(file_path: Path) -> Union[Tuple[int, int], None]:
    """打开图像文件并返回其 (宽度, 高度) 尺寸。"""
    try:
        with Image.open(file_path) as img:
            return img.size
    except FileNotFoundError:
        print(f"警告: 文件未找到 -> {file_path}")
        return None
    except Exception as e:
        print(f"警告: 无法打开或读取文件 -> {file_path}, 错误: {e}")
        return None

def analyze_dataset(json_file_path: Path, project_root: Path):
    """分析单个JSON文件所定义的数据集。"""
    if not json_file_path.exists():
        print(f"\n跳过分析：JSON文件未找到 -> {json_file_path}")
        return None, None, None

    print(f"\n--- 正在分析: {json_file_path.name} ---")

    with open(json_file_path, 'r') as f:
        data_info = json.load(f)

    raw_image_root_str = data_info.get("root_raw_image_dir")
    mask_root_str = data_info.get("root_labeled_mask_dir")

    if not raw_image_root_str:
        print(f"错误: JSON文件 {json_file_path.name} 中缺少 'root_raw_image_dir' 键。")
        return None, None, None
        
    raw_image_root = project_root / raw_image_root_str
    mask_root = project_root / mask_root_str if mask_root_str else None

    samples = data_info.get("samples", [])
    
    image_sizes = defaultdict(int)
    mask_sizes = defaultdict(int)
    mismatched_files = []

    for sample in tqdm(samples, desc="Processing samples"):
        frame_paths = sample.get("frames", [])
        if not frame_paths:
            continue
            
        first_frame_path = raw_image_root / frame_paths[0]
        img_size = get_image_size(first_frame_path)
        if img_size:
            image_sizes[img_size] += len(frame_paths)

        mask_filename = sample.get("mask_file")
        if mask_root and mask_filename:
            mask_path = mask_root / mask_filename
            mask_size = get_image_size(mask_path)
            if mask_size:
                mask_sizes[mask_size] += 1
            
            if img_size and mask_size and img_size != mask_size:
                mismatched_files.append({
                    "image": str(first_frame_path.relative_to(project_root)),
                    "image_size": img_size,
                    "mask": str(mask_path.relative_to(project_root)),
                    "mask_size": mask_size,
                })

    return image_sizes, mask_sizes, mismatched_files


def main():
    """主执行函数。"""
    print("======================================================")
    print("      开始分析数据集中的图像和掩码尺寸      ")
    print("======================================================")
    
    # 路径拼接现在是基于正确的 project_root
    json_dir = project_root / cfg_json_gen.JSON_OUTPUT_DIR_NAME
    
    files_to_analyze = {
        "训练集": json_dir / cfg_train.TRAIN_JSON_NAME,
        "验证集": json_dir / cfg_train.VAL_JSON_NAME,
        "微调训练集": json_dir / cfg_finetune.TRAIN_JSON_NAME,
        "微调验证集": json_dir / cfg_finetune.VAL_JSON_NAME,
    }
    
    all_image_sizes = defaultdict(int)
    all_mask_sizes = defaultdict(int)
    all_mismatched_files = []

    unique_json_paths = {p: name for name, p in files_to_analyze.items()}
    for path_obj in unique_json_paths:
        img_sizes, msk_sizes, mismatched = analyze_dataset(path_obj, project_root)
        
        if img_sizes:
            for size, count in img_sizes.items():
                all_image_sizes[size] += count
        if msk_sizes:
            for size, count in msk_sizes.items():
                all_mask_sizes[size] += count
        if mismatched:
            all_mismatched_files.extend(mismatched)
            
    print("\n\n======================================================")
    print("                  分析报告总结                  ")
    print("======================================================")

    print("\n>>> 图像文件尺寸分布 (宽度, 高度):")
    if not all_image_sizes:
        print("未找到任何图像文件。")
    else:
        for size, count in all_image_sizes.items():
            print(f"  - 尺寸 {size}: 共 {count} 个文件")
            
    print("\n>>> 掩码文件尺寸分布 (宽度, 高度):")
    if not all_mask_sizes:
        print("未找到任何掩码文件。")
    else:
        for size, count in all_mask_sizes.items():
            print(f"  - 尺寸 {size}: 共 {count} 个文件")
            
    print("\n>>> 尺寸不匹配的图像/掩码对:")
    if not all_mismatched_files:
        print("优秀！未发现任何尺寸不匹配的图像/掩码对。")
    else:
        print(f"警告！发现 {len(all_mismatched_files)} 对不匹配的文件：")
        for item in all_mismatched_files:
            print("-" * 30)
            print(f"  图像: {item['image']}")
            print(f"    尺寸: {item['image_size']}")
            print(f"  掩码: {item['mask']}")
            print(f"    尺寸: {item['mask_size']}  <--- 不匹配！")
            
    print("\n======================================================")
    print("                      分析结束                      ")
    print("======================================================")


if __name__ == "__main__":
    main() 