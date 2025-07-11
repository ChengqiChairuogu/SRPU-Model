import json
from pathlib import Path
import albumentations as A
from torchvision import transforms
import numpy as np
from typing import Optional, List, Tuple, Dict, Any

try:
    from configs import base as cfg_base
    from configs import json_config as cfg_json_gen # 用于 JSON_OUTPUT_DIR_NAME
except ImportError:
    # 这个代码块允许脚本在测试时独立运行，
    # 假设了特定的项目结构。
    import sys
    current_script_path = Path(__file__).resolve()
    # 假设的结构: SRPU-Model/utils/augmentations_utils.py
    # 那么项目根目录是 current_script_path.parent.parent
    project_root_for_config = current_script_path.parent.parent
    if str(project_root_for_config) not in sys.path:
        sys.path.insert(0, str(project_root_for_config))
    try:
        from configs import base as cfg_base
        from configs import json_config as cfg_json_gen
    except ImportError as e:
        print(f"错误: 无法从 'configs' 目录导入配置文件。")
        print(f"尝试的项目根目录: {project_root_for_config}")
        print(f"原始 ImportError: {e}")
        # 如果无法加载配置，定义备用方案 (例如，用于隔离测试)
        class MockCfgBase:
            PROJECT_ROOT = Path(".").resolve() # 如果没有更好的选择，默认为当前目录
            IMAGE_HEIGHT = 256
            IMAGE_WIDTH = 256
        class MockCfgJson:
            JSON_OUTPUT_DIR_NAME = "json"
            INPUT_DEPTH = 3 # 默认值
            # STATS_CALCULATION_JSON = "master_labeled_dataset_train.json" # dataset_statistics_calculator.py 中使用
            # STATS_CALC_BATCH_SIZE = 16 # dataset_statistics_calculator.py 中使用


        if 'cfg_base' not in locals(): cfg_base = MockCfgBase()
        if 'cfg_json_gen' not in locals(): cfg_json_gen = MockCfgJson()
        print("警告: 由于主配置导入失败，正在使用模拟配置。")



def load_dataset_stats(
    stats_json_name: str = "dataset_stats.json",
    project_root: Optional[Path] = None,
    json_dir_name_relative_to_project: Optional[str] = None,
    expected_input_depth: Optional[int] = None
) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """
    从 dataset_stats.json 文件加载均值和标准差。

    Args:
        stats_json_name (str): 包含统计数据的JSON文件名。
        project_root (Optional[Path]): 项目根目录的绝对路径。如果为None，则使用 cfg_base.PROJECT_ROOT。
        json_dir_name_relative_to_project (Optional[str]): JSON目录相对于项目根目录的路径。
                                                        如果为None，则使用 cfg_json_gen.JSON_OUTPUT_DIR_NAME。
        expected_input_depth (Optional[int]): 期望的输入深度。用于验证加载的统计数据是否匹配。
                                              如果为None，则使用 cfg_json_gen.INPUT_DEPTH。

    Returns:
        Tuple[Optional[List[float]], Optional[List[float]]]: (mean, std) 或 (None, None) 如果失败。
    """
    if project_root is None:
        project_root = cfg_base.PROJECT_ROOT.resolve()
    if json_dir_name_relative_to_project is None:
        json_dir_name_relative_to_project = cfg_json_gen.JSON_OUTPUT_DIR_NAME

    stats_file_path = project_root / json_dir_name_relative_to_project / stats_json_name

    if not stats_file_path.exists():
        print(f"警告: 统计文件未找到: {stats_file_path}")
        print("请运行 'utils/dataset_statistics_calculator.py' 来生成它。")
        print("将不使用均值和标准差进行归一化。")
        return None, None

    try:
        with open(stats_file_path, 'r') as f:
            stats_data = json.load(f)
        
        mean = stats_data.get("mean")
        std = stats_data.get("std")
        input_depth_at_calculation = stats_data.get("input_depth_at_calculation")

        if expected_input_depth is None:
            expected_input_depth = cfg_json_gen.INPUT_DEPTH
        
        if input_depth_at_calculation is not None and expected_input_depth is not None:
            if input_depth_at_calculation != expected_input_depth:
                print(f"警告: 统计文件中的 input_depth ({input_depth_at_calculation}) "
                      f"与期望的 input_depth ({expected_input_depth}) 不匹配。")
                print(f"统计数据可能不准确。文件: {stats_file_path}")
                # 根据您的策略，您可以选择在此处返回 None, None
                # return None, None 
            if mean and len(mean) != expected_input_depth:
                print(f"警告: 统计文件中的均值长度 ({len(mean)}) "
                      f"与期望的 input_depth ({expected_input_depth}) 不匹配。")
                return None, None
            if std and len(std) != expected_input_depth:
                print(f"警告: 统计文件中的标准差长度 ({len(std)}) "
                      f"与期望的 input_depth ({expected_input_depth}) 不匹配。")
                return None, None


        if mean is not None and std is not None:
            print(f"从 {stats_file_path} 加载的均值: {mean}")
            print(f"从 {stats_file_path} 加载的标准差: {std}")
            return mean, std
        else:
            print(f"警告: 未能从 {stats_file_path} 加载均值或标准差。")
            return None, None

    except Exception as e:
        print(f"加载统计文件 {stats_file_path} 时出错: {e}")
        return None, None

def get_normalization_transform(
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
    load_from_file: bool = True,
    stats_json_name: str = "dataset_stats.json",
    project_root: Optional[Path] = None,
    json_dir_name_relative_to_project: Optional[str] = None,
    input_depth_for_stats_check: Optional[int] = None
) -> Optional[transforms.Normalize]:
    """
    获取PyTorch的归一化变换。
    可以从文件加载统计数据，或者直接传递均值和标准差。

    Args:
        mean (Optional[List[float]]): 用户提供的均值。
        std (Optional[List[float]]): 用户提供的标准差。
        load_from_file (bool): 如果为True且mean/std未提供，则尝试从文件加载。
        stats_json_name (str): 统计文件名。
        project_root (Optional[Path]): 项目根目录。
        json_dir_name_relative_to_project (Optional[str]): JSON目录的相对路径。
        input_depth_for_stats_check (Optional[int]): 用于检查统计文件一致性的输入深度。

    Returns:
        Optional[transforms.Normalize]: PyTorch归一化变换，如果统计数据可用。
    """
    if mean is None or std is None:
        if load_from_file:
            if input_depth_for_stats_check is None:
                # 尝试从配置中获取默认的 input_depth
                try:
                    input_depth_for_stats_check = cfg_json_gen.INPUT_DEPTH
                except AttributeError:
                    print("警告: 无法从配置中获取 cfg_json_gen.INPUT_DEPTH。统计检查可能不完整。")
                    input_depth_for_stats_check = None # 明确设置为None

            loaded_mean, loaded_std = load_dataset_stats(
                stats_json_name=stats_json_name,
                project_root=project_root,
                json_dir_name_relative_to_project=json_dir_name_relative_to_project,
                expected_input_depth=input_depth_for_stats_check
            )
            if loaded_mean is not None and loaded_std is not None:
                mean, std = loaded_mean, loaded_std
            else:
                return None # 无法加载统计数据
        else:
            print("警告: 未提供均值/标准差，并且未启用从文件加载。不应用归一化。")
            return None
    
    if mean is not None and std is not None:
        # 确保std中的值不为零，以避免除以零的错误
        std_processed = [max(s, 1e-6) for s in std]
        return transforms.Normalize(mean=mean, std=std_processed)
    return None


def get_train_augmentations(
    height: int, 
    width: int,
    additional_targets: Optional[Dict[str, str]] = None
) -> A.Compose:
    """
    为训练集获取Albumentations增强流水线。
    这些是在NumPy数组上操作的增强，不包括ToTensor或Normalize。

    Args:
        height (int): 图像高度。
        width (int): 图像宽度。
        additional_targets (Optional[Dict[str, str]]): 传递给 A.Compose 的 additional_targets。
                                                        例如: {"mask1": "mask", "mask2": "mask"}

    Returns:
        A.Compose: Albumentations增强流水线。
    """
    if additional_targets is None:
        additional_targets = {}

    return A.Compose([
        A.PadIfNeeded(min_height=height, min_width=width, border_mode=0, value=0, mask_value=0, p=1.0), # value for image, mask_value for mask
        A.RandomCrop(height=height, width=width, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5), # 交换x和y轴
        
        # 几何失真
        A.OneOf([
            A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=0.5) # 增加 distort_limit
        ], p=0.3), #降低这个OneOf的概率

        # 颜色/亮度增强 (对灰度图像可能意义不大，但如果输入是RGB或多通道可以保留)
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.7), # 对灰度图影响小
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5), # CLAHE对SEM可能有用
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5), # 高斯噪声
        ], p=0.5), #降低这个OneOf的概率
        
        #模糊
        A.OneOf([
            A.MotionBlur(p=0.3),
            A.MedianBlur(blur_limit=3, p=0.3),
            A.Blur(blur_limit=3, p=0.3),
        ], p=0.3), #降低这个OneOf的概率

        # Dropout类增强
        # A.CoarseDropout(max_holes=8, max_height=height//10, max_width=width//10, 
        #                 min_holes=2, min_height=height//20, min_width=width//20, 
        #                 fill_value=0, mask_fill_value=0, p=0.3),

    ], additional_targets=additional_targets if additional_targets else {"mask": "mask"}) # 默认目标是 "mask"


def get_val_augmentations(
    height: int, 
    width: int,
    additional_targets: Optional[Dict[str, str]] = None
) -> A.Compose:
    """
    为验证/测试集获取Albumentations增强流水线。
    通常只包含必要的调整大小/填充。

    Args:
        height (int): 图像高度。
        width (int): 图像宽度。
        additional_targets (Optional[Dict[str, str]]): 传递给 A.Compose 的 additional_targets。

    Returns:
        A.Compose: Albumentations增强流水线。
    """
    if additional_targets is None:
        additional_targets = {}
    # 对于验证，通常我们只需要确保输入尺寸正确，例如通过PadIfNeeded或CenterCrop
    # 如果训练时用了RandomCrop，验证时可能用CenterCrop或确保完整图像评估
    return A.Compose([
        A.PadIfNeeded(min_height=height, min_width=width, border_mode=0, value=0, mask_value=0, p=1.0),
        A.CenterCrop(height=height, width=width, p=1.0),
    ], additional_targets=additional_targets if additional_targets else {"mask": "mask"})


# --- 测试 ---
if __name__ == '__main__':
    print("--- 测试 augmentations_utils.py ---")

    # 模拟配置值 (如果导入失败，会使用MockCfg)
    try:
        IMG_H = cfg_base.IMAGE_HEIGHT
        IMG_W = cfg_base.IMAGE_WIDTH
        INPUT_D = cfg_json_gen.INPUT_DEPTH
        PROJECT_R = cfg_base.PROJECT_ROOT.resolve()
        JSON_DIR_NAME = cfg_json_gen.JSON_OUTPUT_DIR_NAME
    except AttributeError as e:
        print(f"配置属性缺失，无法运行完整测试: {e}")
        print("请确保 configs/base.py 和 configs/json_config.py 已正确配置。")
        IMG_H, IMG_W, INPUT_D, PROJECT_R, JSON_DIR_NAME = 256, 256, 3, Path("."), "json"
        print(f"使用默认值进行部分测试: H={IMG_H}, W={IMG_W}, D={INPUT_D}")


    # 1. 测试加载统计数据
    print("\n1. 测试加载统计数据...")
    # 为了测试，可以手动在 PROJECT_R / JSON_DIR_NAME / "dataset_stats.json" 创建一个假的统计文件

    # 确保json目录存在
    json_output_dir_for_test = PROJECT_R / JSON_DIR_NAME
    json_output_dir_for_test.mkdir(parents=True, exist_ok=True)
    
    # 创建一个临时的假的 dataset_stats.json 用于测试
    dummy_stats_path = json_output_dir_for_test / "dummy_test_stats.json"
    if INPUT_D > 0 : # 仅当INPUT_D有效时创建
        with open(dummy_stats_path, 'w') as f_stats:
            json.dump({
                "mean": [0.485] * INPUT_D,
                "std": [0.229] * INPUT_D,
                "input_depth_at_calculation": INPUT_D,
                "source_json": "test_dummy.json"
            }, f_stats, indent=4)
        print(f"创建了假的统计文件: {dummy_stats_path}")

        mean, std = load_dataset_stats(
            stats_json_name="dummy_test_stats.json",
            project_root=PROJECT_R,
            json_dir_name_relative_to_project=JSON_DIR_NAME,
            expected_input_depth=INPUT_D
        )
        if mean and std:
            print(f"  成功加载均值: {mean}, 标准差: {std}")
        else:
            print("  未能加载统计数据。")

        # 2. 测试获取归一化变换
        print("\n2. 测试获取归一化变换...")
        norm_transform = get_normalization_transform(
            load_from_file=True, # 强制从文件加载
            stats_json_name="dummy_test_stats.json",
            project_root=PROJECT_R,
            json_dir_name_relative_to_project=JSON_DIR_NAME,
            input_depth_for_stats_check=INPUT_D
        )
        if norm_transform:
            print(f"  成功获取归一化变换: {norm_transform}")
            # 测试变换 (需要一个假的张量)
            try:
                dummy_tensor = torch.rand(INPUT_D, IMG_H, IMG_W)
                transformed_tensor = norm_transform(dummy_tensor)
                print(f"    归一化前均值 (近似): {[dummy_tensor[c].mean().item() for c in range(INPUT_D)]}")
                print(f"    归一化后均值 (近似): {[transformed_tensor[c].mean().item() for c in range(INPUT_D)]}") # 应该接近0
            except Exception as e:
                 print(f"    测试归一化变换时出错: {e}")
        else:
            print("  未能获取归一化变换。")
        
        # 清理假的统计文件
        if dummy_stats_path.exists():
            dummy_stats_path.unlink()

    else:
        print("INPUT_D 为0或无效，跳过依赖于INPUT_D的统计加载和归一化测试。")


    # 3. 测试获取训练增强
    print("\n3. 测试获取训练增强...")
    train_augs = get_train_augmentations(height=IMG_H, width=IMG_W)
    if train_augs:
        print(f"  成功获取训练增强: {train_augs}")
        # 测试增强 (需要假的numpy图像和掩码)
        if INPUT_D > 0:
            dummy_image_np = np.random.randint(0, 256, (IMG_H, IMG_W, INPUT_D), dtype=np.uint8)
            dummy_masks_list_np = [np.random.randint(0, 2, (IMG_H, IMG_W), dtype=np.uint8) for _ in range(INPUT_D)]
            
            try:
                augmented = train_augs(image=dummy_image_np, masks=dummy_masks_list_np)
                aug_image = augmented['image']
                aug_masks = augmented['masks']
                print(f"    原始图像形状: {dummy_image_np.shape}, 增强后图像形状: {aug_image.shape}")
                print(f"    原始掩码列表长度: {len(dummy_masks_list_np)}, 增强后掩码列表长度: {len(aug_masks)}")
                if aug_masks:
                    print(f"    增强后单个掩码形状: {aug_masks[0].shape}")
            except Exception as e:
                print(f"    测试训练增强时出错: {e}")
        else:
            print("    INPUT_D为0，跳过训练增强的NumPy数组测试。")


    # 4. 测试获取验证增强
    print("\n4. 测试获取验证增强...")
    val_augs = get_val_augmentations(height=IMG_H, width=IMG_W)
    if val_augs:
        print(f"  成功获取验证增强: {val_augs}")
        if INPUT_D > 0:
            dummy_image_np = np.random.randint(0, 256, (IMG_H + 20, IMG_W + 20, INPUT_D), dtype=np.uint8) # 比目标大一点
            dummy_masks_list_np = [np.random.randint(0, 2, (IMG_H + 20, IMG_W + 20), dtype=np.uint8) for _ in range(INPUT_D)]
            try:
                augmented = val_augs(image=dummy_image_np, masks=dummy_masks_list_np)
                aug_image = augmented['image']
                aug_masks = augmented['masks']
                print(f"    原始图像形状: {dummy_image_np.shape}, 增强后图像形状: {aug_image.shape}") # 应该被裁剪/填充到 IMG_H, IMG_W
                assert aug_image.shape == (IMG_H, IMG_W, INPUT_D)
                if aug_masks:
                     assert aug_masks[0].shape == (IMG_H, IMG_W)

            except Exception as e:
                print(f"    测试验证增强时出错: {e}")
        else:
            print("    INPUT_D为0，跳过验证增强的NumPy数组测试。")

    print("\n--- augmentations_utils.py 测试完成 ---")