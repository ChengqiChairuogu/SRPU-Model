import os
import json
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Callable, Dict, Any, Tuple
from torchvision import transforms
import sys

try:
    from configs import base as cfg_base
    from configs import json_config as cfg_json_gen
    from utils import augmentation
except ImportError:
    if __name__ == '__main__' and __package__ is None:
        current_script_path = Path(__file__).resolve()
        project_root_for_import = current_script_path.parent.parent 
        
        if str(project_root_for_import) not in sys.path:
            sys.path.insert(0, str(project_root_for_import))
        try:
            from configs import base as cfg_base
            from configs import json_config as cfg_json_gen
            from utils import augmentations_utils
        except ImportError as e_inner:
            print(f"错误: 无法导入所需模块: {e_inner}")
            print(f"尝试的项目根目录: {project_root_for_import}")
            # 为测试提供备用导入
            if 'cfg_base' not in locals(): 
                class MockCfgBase: PROJECT_ROOT = Path(".").resolve(); IMAGE_HEIGHT=64;IMAGE_WIDTH=64; MAPPING={(0,0,0):0, (255,255,255):1}; NUM_CLASSES=2
                cfg_base = MockCfgBase()
                print("警告: 使用 MockCfgBase")
            if 'cfg_json_gen' not in locals():
                class MockCfgJson: JSON_OUTPUT_DIR_NAME="json"; RAW_IMAGE_SOURCE_DIR_NAME="data/raw"; LABELED_MASK_DIR_NAME="data/labeled"; INPUT_DEPTH=3
                cfg_json_gen = MockCfgJson()
                print("警告: 使用 MockCfgJson")
            if 'augmentations_utils' not in locals():
                class MockAugUtils:
                    def get_train_augmentations(self, *args, **kwargs): return None
                    def get_val_augmentations(self, *args, **kwargs): return None
                    def get_normalization_transform(self, *args, **kwargs): return None
                    def load_dataset_stats(self, *args, **kwargs): return None, None
                augmentations_utils = MockAugUtils()
                print("警告: 使用 MockAugUtils")
    else:
        raise


class SemSegmentationDataset(Dataset):
    def __init__(self,
                 json_file_identifier: str,
                 project_root: Path,
                 input_depth_from_config: int,
                 class_mapping_from_config: Dict[Tuple[int, int, int], int],
                 json_dir_name_relative_to_project: str,
                 augmentations: Optional[Callable] = None, # Albumentations Compose 对象
                 normalization_transform: Optional[Callable] = None): # PyTorch Normalize 对象

        self.project_root = project_root.resolve()
        self.input_depth = input_depth_from_config
        self.class_mapping = class_mapping_from_config
        self.augmentations = augmentations
        self.normalization_transform = normalization_transform

        json_file_path_obj = Path(json_file_identifier)
        if json_file_path_obj.is_absolute():
            self.json_file_abs_path = json_file_path_obj
        else:
            potential_path_as_is = self.project_root / json_file_identifier
            if potential_path_as_is.exists() and json_dir_name_relative_to_project in json_file_identifier:
                 self.json_file_abs_path = potential_path_as_is
            else:
                 self.json_file_abs_path = self.project_root / json_dir_name_relative_to_project / json_file_identifier
        
        if not self.json_file_abs_path.exists():
            alt_path = self.project_root / json_dir_name_relative_to_project / Path(json_file_identifier).name
            if alt_path.exists():
                self.json_file_abs_path = alt_path
            else:
                raise FileNotFoundError(f"解析的JSON文件未找到。已检查: {self.json_file_abs_path} 和 {alt_path}")

        try:
            with open(self.json_file_abs_path, 'r') as f:
                self.data_info = json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"从以下位置解码JSON时出错: {self.json_file_abs_path}")

        self.samples = self.data_info.get("samples", [])
        if not self.samples:
            print(f"警告: 在 {self.json_file_abs_path} 中未找到样本。")

        self.raw_image_root_name_in_json = self.data_info.get("root_raw_image_dir")
        self.mask_root_name_in_json = self.data_info.get("root_labeled_mask_dir")

        if not self.raw_image_root_name_in_json:
            raise ValueError(f"在JSON文件中未找到 'root_raw_image_dir': {self.json_file_abs_path}")

        json_input_depth = self.data_info.get("input_depth")
        if json_input_depth is not None and json_input_depth != self.input_depth:
            print(f"警告: 来自JSON的 input_depth ({json_input_depth}) 与 "
                  f"配置/参数 ({self.input_depth}) 不同。使用参数中的 {self.input_depth}。")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image_frame(self, frame_relative_path: str) -> np.ndarray:
        """
        加载单个图像帧，将其转换为 float32 NumPy 数组，并缩放到 [0, 1] 范围。
        输出图像的维度为 (H, W) 或 (H, W, 3)。
        """
        image_path = self.project_root / self.raw_image_root_name_in_json / frame_relative_path
        try:
            img_pil = Image.open(image_path)

            # 统一转换为 L (灰度) 或 RGB
            if img_pil.mode == 'RGBA':
                img_pil = img_pil.convert('RGB')
            elif img_pil.mode == 'LA': # 带alpha的灰度
                img_pil = img_pil.convert('L')
            elif img_pil.mode == 'P': # 调色板图像，转换为RGB或L
                # 尝试转换为RGB，如果失败（例如，没有足够的颜色信息），则转换为L
                try:
                    img_pil = img_pil.convert('RGB')
                except ValueError:
                    img_pil = img_pil.convert('L')
            elif img_pil.mode == '1': # 二值图像
                 img_pil = img_pil.convert('L')
            elif img_pil.mode not in ['L', 'RGB', 'I', 'F']: # 其他PIL支持的模式，如CMYK, YCbCr
                # 对于 'I' (整数) 和 'F' (浮点数) 模式，如果它们已经是单通道且范围合适，
                # np.array 可能可以直接工作。但为了与 ImageNet 预训练模型兼容（通常期望0-1输入），
                # 转换为 L 或 RGB 然后缩放是更安全的方式。
                # 这里我们选择转换为 L，因为这是原始代码的倾向。
                print(f"警告: 图像 {image_path} 的模式为 {img_pil.mode}，将尝试转换为 'L'。")
                img_pil = img_pil.convert('L')
            
            # 对于 'I' 和 'F' 模式，如果它们未在上面转换，我们需要特别处理
            if img_pil.mode == 'F': # 32位浮点像素
                img_np = np.array(img_pil, dtype=np.float32)
                # 假设 'F' 模式图像的像素值已经在 [0,1] 范围内，或者需要特定的归一化逻辑
                # 如果不是，这里需要添加额外的缩放。为简单起见，我们假设它们已经是[0,1]
                # 或者，更好的做法是也将其转换为L/RGB并从uint8缩放，除非有特定理由保留原始float范围
            elif img_pil.mode == 'I': # 32位有符号整数像素
                img_np = np.array(img_pil, dtype=np.int32)
                # 'I' 模式的范围可能很大。将其安全地转换为[0,1] float32需要了解其原始范围。
                # 一个简单的（但可能有损的）方法是将其视为16位图像处理（如果适用）或转换为L
                # 这里为了代码的简单性并与之前的逻辑一致，我们假设这类图像如果未被明确转换，
                # 其处理方式应该在下游（如模型本身）或通过更具体的配置来定义。
                # 目前，如果上面没有 convert('L')，这里的类型转换可能不够。
                # 最安全的做法是，如果img_pil.mode仍然是 'I' 或 'F'，但我们期望[0,1]的输入，
                # 应该再次尝试转换或发出更强的警告/错误。
                # 让我们坚持上面的转换逻辑：未知模式（包括I,F如果不在白名单）会转为L。
                # 如果它就是L或RGB，则np.array()后是uint8。
                img_np = np.array(img_pil).astype(np.float32) / 255.0

            else: # 应该是 L 或 RGB 模式了
                img_np = np.array(img_pil) # 对于 L/RGB 通常是 uint8
                if img_np.dtype == np.uint8:
                    img_np = img_np.astype(np.float32) / 255.0
                elif img_np.dtype == np.uint16: # 例如 16-bit TIFF
                    img_np = img_np.astype(np.float32) / 65535.0
                elif img_np.dtype != np.float32: # 如果是其他类型，尝试转换为float32
                    img_np = img_np.astype(np.float32)
                    # 这里可能需要根据具体数据类型进行范围检查或缩放

            # 确保图像是 HxW 或 HxWx3
            if img_np.ndim == 2: # (H, W)
                pass # Albumentations 可以处理 HWC, 所以单通道图像后面会扩展维度
            elif img_np.ndim == 3 and img_np.shape[-1] == 1: # (H, W, 1)
                pass
            elif img_np.ndim == 3 and img_np.shape[-1] == 3: # (H, W, 3)
                pass
            else:
                # 对于不常见的通道数，取第一个通道并使其成为 (H, W)
                print(f"警告: 图像 {image_path} (模式: {img_pil.mode}) 加载后形状为 {img_np.shape}。"
                      "将尝试使用第一个通道并调整为2D或3D (H,W,1)。")
                if img_np.ndim == 3 and img_np.shape[-1] > 1:
                    img_np = img_np[..., 0] # 取第一个通道
                # 确保结果至少是2D
                if img_np.ndim < 2 :
                    raise ValueError(f"图像 {image_path} 加载和处理后维度过低: {img_np.shape}")


            return img_np

        except FileNotFoundError:
            raise FileNotFoundError(f"图像帧未找到: {image_path}")
        except Exception as e:
            raise IOError(f"打开或处理图像 {image_path} 时出错: {e}")


    def _load_and_convert_mask(self, mask_relative_path: str) -> Optional[np.ndarray]:
        if not mask_relative_path or not self.mask_root_name_in_json:
            return None

        mask_path = self.project_root / self.mask_root_name_in_json / mask_relative_path
        try:
            mask_pil = Image.open(mask_path)
        except FileNotFoundError:
            return None

        mask_np_original = np.array(mask_pil)

        if mask_np_original.ndim == 2: # 已经是类别索引掩码 (H, W)
            return mask_np_original.astype(np.uint8)
        elif mask_np_original.ndim == 3 and self.class_mapping: # RGB 掩码 (H, W, C)
            if mask_np_original.shape[-1] == 4: # RGBA
                mask_np_original = mask_np_original[..., :3] # 取 RGB
            if mask_np_original.shape[-1] != 3:
                raise ValueError(f"掩码 {mask_path} 是3维但非RGB (形状: {mask_np_original.shape})。无法应用RGB class_mapping。")

            mask_np_labels = np.zeros((mask_np_original.shape[0], mask_np_original.shape[1]), dtype=np.uint8)
            for color_rgb_tuple, class_idx in self.class_mapping.items():
                match = np.all(mask_np_original == np.array(color_rgb_tuple, dtype=np.uint8), axis=-1)
                mask_np_labels[match] = class_idx
            return mask_np_labels
        elif mask_np_original.ndim == 3 and not self.class_mapping: # 假设是 (H,W,1) 形式的类别索引
            return mask_np_original[..., 0].astype(np.uint8)
        else:
            raise ValueError(f"不支持的掩码格式或维度: {mask_np_original.shape} 对于掩码 {mask_path}")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.samples:
            raise IndexError("数据集为空或样本未正确加载。")
        if idx >= len(self.samples):
            raise IndexError("索引越界")

        sample_info = self.samples[idx]
        image_frame_paths_in_json: List[str] = sample_info["frames"]
        mask_file_path_in_json: Optional[str] = sample_info.get("mask_file")

        # 1. 加载并堆叠图像帧 (结果为 H, W, C_stacked)
        img_frames_list_np: List[np.ndarray] = []
        for frame_relative_path in image_frame_paths_in_json:
            img_np = self._load_image_frame(frame_relative_path) # 返回 (H,W) 或 (H,W,3) float32 [0,1]
            if img_np.ndim == 2:
                img_np = np.expand_dims(img_np, axis=-1) # 转换为 (H, W, 1)
            elif img_np.ndim == 3 and img_np.shape[-1] != 1 and img_np.shape[-1] != 3:
                 # 这个逻辑可能需要重新审视，_load_image_frame 应该已经处理了通道
                 # 但作为安全措施保留
                if img_np.shape[-1] > 1:
                    print(f"警告: 帧 {frame_relative_path} 有 {img_np.shape[-1]} 个通道，但不是1或3。取第一个通道。")
                    img_np = img_np[..., [0]] # (H, W, 1)
            img_frames_list_np.append(img_np)
        
        try:
            # 确保所有帧具有相同的 H, W
            first_shape = img_frames_list_np[0].shape[:2]
            for i, frame in enumerate(img_frames_list_np):
                if frame.shape[:2] != first_shape:
                    # 尝试调整大小 (或者抛出错误)
                    # print(f"警告: 样本 {sample_info.get('sample_id', idx)} 中的帧 {i} 形状 {frame.shape[:2]} 与第一帧 {first_shape} 不同。将尝试调整大小。")
                    # pil_frame = Image.fromarray((frame * 255).astype(np.uint8).squeeze()) # 转回PIL以便调整大小
                    # pil_frame_resized = pil_frame.resize((first_shape[1], first_shape[0]), Image.BILINEAR)
                    # img_frames_list_np[i] = (np.array(pil_frame_resized).astype(np.float32) / 255.0)[..., np.newaxis]
                    raise ValueError(f"样本 {sample_info.get('sample_id', idx)} 中的帧 {i} 形状 {frame.shape[:2]} 与第一帧 {first_shape} 不同。请确保所有帧具有相同的空间维度。")


            stacked_images_np = np.concatenate(img_frames_list_np, axis=-1).astype(np.float32) # (H, W, D*num_img_channels)
        except ValueError as e:
            print(f"连接图像帧时出错，样本ID: {sample_info.get('sample_id', 'N/A')}")
            print(f"路径: {image_frame_paths_in_json}")
            print(f"加载帧的形状 (连接前): {[img.shape for img in img_frames_list_np]}")
            raise e

        # stacked_images_np 的最后一个维度应该是 self.input_depth * (1 if grayscale else 3)
        # Albumentations 期望输入图像为 (H, W, C)
        # 这里的 C 是 stacked_images_np.shape[-1]

        # 2. 加载并处理掩码 (结果为 (D, H, W) np.uint8 或 np.int64)
        h_ref, w_ref = stacked_images_np.shape[:2]
        mask_list_for_aug: List[np.ndarray] = [] # 用于传递给Albumentations的掩码列表

        center_mask_np: Optional[np.ndarray] = None
        if mask_file_path_in_json:
            center_mask_np = self._load_and_convert_mask(mask_file_path_in_json) # (H, W) uint8
            if center_mask_np is None: # 如果文件存在但加载失败
                center_mask_np = np.zeros((h_ref, w_ref), dtype=np.uint8)
                print(f"警告: 无法加载掩码 {mask_file_path_in_json}，使用空掩码。")
            elif center_mask_np.shape != (h_ref, w_ref):
                # print(f"警告: 掩码 {mask_file_path_in_json} 形状 {center_mask_np.shape} 与图像 { (h_ref, w_ref)} 不同。将尝试调整大小。")
                # pil_mask = Image.fromarray(center_mask_np)
                # pil_mask_resized = pil_mask.resize((w_ref, h_ref), Image.NEAREST) # 掩码用NEAREST
                # center_mask_np = np.array(pil_mask_resized)
                 raise ValueError(f"掩码 {mask_file_path_in_json} 形状 {center_mask_np.shape} 与图像 {(h_ref, w_ref)} 不同。")
        else: # 无掩码文件路径（例如，对于无标签数据）
            center_mask_np = np.zeros((h_ref, w_ref), dtype=np.uint8) # 创建一个虚拟的空掩码

        # 根据 input_depth 复制中心掩码 self.input_depth 次
        # 这假设我们希望对每个输入帧/通道应用相同的目标掩码
        for _ in range(self.input_depth):
            mask_list_for_aug.append(center_mask_np.copy())


        # 3. 应用数据增强 (Albumentations)
        # stacked_images_np 已经是 (H, W, C_stacked) float32 [0,1]
        # mask_list_for_aug 是一个包含 D 个 (H, W) uint8 数组的列表
        if self.augmentations:
            try:
                # Albumentations 的 'masks' 参数期望一个掩码列表
                augmented_data = self.augmentations(image=stacked_images_np, masks=mask_list_for_aug)
                stacked_images_np = augmented_data['image']
                augmented_masks_list = augmented_data['masks']
            except Exception as e:
                print(f"样本 {sample_info.get('sample_id', idx)} 的增强过程中出错: {e}")
                # 如果增强失败，保持原始数据（或根据需要处理错误）
                augmented_masks_list = mask_list_for_aug 
                # raise e # 重新抛出错误，以便调试
        else:
            augmented_masks_list = mask_list_for_aug

        # 将增强后的掩码列表转换回 (D, H, W) NumPy 数组
        # 确保所有掩码在增强后具有相同的形状
        if augmented_masks_list:
            final_mask_h, final_mask_w = augmented_masks_list[0].shape
            processed_masks_for_stack = []
            for m_idx, m_aug in enumerate(augmented_masks_list):
                if m_aug.shape != (final_mask_h, final_mask_w):
                    # 这不应该发生，如果Albumentations配置正确的话
                    print(f"警告: 增强后的掩码 {m_idx} 形状 {m_aug.shape} 与第一个掩码 {(final_mask_h, final_mask_w)} 不同。")
                    # 可以尝试调整大小或填充，但最好是确保增强配置正确
                    # 这里我们简单地使用第一个掩码的形状进行裁剪或填充 (这可能不是最佳做法)
                    pil_m_aug = Image.fromarray(m_aug)
                    pil_m_resized = pil_m_aug.resize((final_mask_w, final_mask_h), Image.NEAREST)
                    processed_masks_for_stack.append(np.array(pil_m_resized))
                else:
                    processed_masks_for_stack.append(m_aug)
            stacked_masks_np = np.stack(processed_masks_for_stack, axis=0).astype(np.int64)
        else: # 如果没有掩码（例如，列表为空）
             # 创建一个符合图像形状的空掩码堆栈
            final_img_h, final_img_w = stacked_images_np.shape[:2]
            stacked_masks_np = np.zeros((self.input_depth, final_img_h, final_img_w), dtype=np.int64)


        # 4. 转换为张量
        # stacked_images_np: (H, W, C_stacked) -> image_tensor: (C_stacked, H, W)
        image_tensor = torch.from_numpy(stacked_images_np.transpose((2, 0, 1))).contiguous().float()
        
        # stacked_masks_np: (D, H, W) -> mask_tensor: (D, H, W)
        mask_tensor = torch.from_numpy(stacked_masks_np).contiguous().long()

        # 5. 应用归一化 (如果提供)
        if self.normalization_transform:
            image_tensor = self.normalization_transform(image_tensor)
            
        return image_tensor, mask_tensor


# ======================================================================
# TEST / DEBUG
# ======================================================================
if __name__ == '__main__':
    print("--- 直接运行 SemSegmentationDataset 进行测试 ---")

    # --- 1. 使用 (或创建最小的) 配置 ---
    # 确保测试时可以访问配置或使用Mock对象
    try:
        current_project_root_for_test = cfg_base.PROJECT_ROOT.resolve()
        json_dir_name_for_test = cfg_json_gen.JSON_OUTPUT_DIR_NAME
        raw_dir_name_for_test = cfg_json_gen.RAW_IMAGE_SOURCE_DIR_NAME
        labeled_dir_name_for_test = cfg_json_gen.LABELED_MASK_DIR_NAME
        test_input_depth = cfg_json_gen.INPUT_DEPTH
        test_img_h = getattr(cfg_base, 'IMAGE_HEIGHT', 64)
        test_img_w = getattr(cfg_base, 'IMAGE_WIDTH', 64)
        test_class_mapping = getattr(cfg_base, 'MAPPING', {(0,0,0):0, (255,0,0):1}) # 示例映射
        num_classes_test = getattr(cfg_base, 'NUM_CLASSES', 2)

    except AttributeError as e:
        print(f"错误: 测试所需的配置属性缺失: {e}")
        print("请确保 configs/base.py 和 configs/json_config.py 已正确配置并可导入。")
        print("将使用在脚本顶部定义的Mock配置进行测试。")
        current_project_root_for_test = cfg_base.PROJECT_ROOT # 来自Mock
        json_dir_name_for_test = cfg_json_gen.JSON_OUTPUT_DIR_NAME
        raw_dir_name_for_test = cfg_json_gen.RAW_IMAGE_SOURCE_DIR_NAME
        labeled_dir_name_for_test = cfg_json_gen.LABELED_MASK_DIR_NAME
        test_input_depth = cfg_json_gen.INPUT_DEPTH
        test_img_h = cfg_base.IMAGE_HEIGHT
        test_img_w = cfg_base.IMAGE_WIDTH
        test_class_mapping = cfg_base.MAPPING
        num_classes_test = cfg_base.NUM_CLASSES


    json_dir_abs_for_test = current_project_root_for_test / json_dir_name_for_test
    raw_dir_abs_for_test = current_project_root_for_test / raw_dir_name_for_test
    labeled_dir_abs_for_test = current_project_root_for_test / labeled_dir_name_for_test

    json_dir_abs_for_test.mkdir(parents=True, exist_ok=True)
    raw_dir_abs_for_test.mkdir(parents=True, exist_ok=True)
    labeled_dir_abs_for_test.mkdir(parents=True, exist_ok=True)

    # --- 2. 创建一个用于测试的伪JSON文件 ---
    dummy_json_filename = "test_debug_sem_dataset.json"
    dummy_json_path = json_dir_abs_for_test / dummy_json_filename
    
    # 确保测试图像尺寸较小以加快测试速度
    dummy_img_h_actual = min(test_img_h, 64) 
    dummy_img_w_actual = min(test_img_w, 64)

    dummy_samples_data_for_json = []
    seq_name_test = "debug_sem_seq1"
    # 在配置的原始图像和标签掩码的根目录下创建序列子目录
    raw_seq_dir_abs = raw_dir_abs_for_test / seq_name_test
    labeled_seq_dir_abs = labeled_dir_abs_for_test / seq_name_test
    raw_seq_dir_abs.mkdir(parents=True, exist_ok=True)
    labeled_seq_dir_abs.mkdir(parents=True, exist_ok=True)

    frame_paths_for_json_sample = []
    mask_path_for_json_sample = None

    # 创建一些伪图像帧 (例如，比input_depth多几帧以测试边界处理)
    total_dummy_frames_in_raw = test_input_depth + 2 
    for frame_idx in range(total_dummy_frames_in_raw):
        frame_basename = f"{seq_name_test}_frame{frame_idx:03d}.png"
        # JSON中的路径是相对于 raw_image_root_name_in_json 的
        dummy_image_rel_path_in_json = Path(seq_name_test) / frame_basename
        dummy_image_abs_path = raw_seq_dir_abs / frame_basename # 绝对路径用于保存文件
        
        if not dummy_image_abs_path.exists():
            # 创建灰度图像 (H, W)
            Image.new('L', (dummy_img_w_actual, dummy_img_h_actual), color='gray').save(dummy_image_abs_path)

    # 选择一个中心帧为其创建掩码
    center_frame_actual_idx = 1 # 这是原始序列中的索引 (0, 1, 2, 3, 4...)
    if center_frame_actual_idx < total_dummy_frames_in_raw:
        center_frame_basename = f"{seq_name_test}_frame{center_frame_actual_idx:03d}.png"
        # JSON中的掩码路径是相对于 mask_root_name_in_json 的
        dummy_mask_rel_path_in_json = Path(seq_name_test) / center_frame_basename 
        dummy_mask_abs_path = labeled_seq_dir_abs / center_frame_basename # 绝对路径用于保存文件
        
        if not dummy_mask_abs_path.exists():
             mask_array = np.zeros((dummy_img_h_actual, dummy_img_w_actual), dtype=np.uint8)
             mask_array[dummy_img_h_actual//4 : dummy_img_h_actual//2, 
                        dummy_img_w_actual//4 : dummy_img_w_actual//2] = 1 # 假设类别1
             Image.fromarray(mask_array, mode='L').save(dummy_mask_abs_path)
        mask_path_for_json_sample = str(dummy_mask_rel_path_in_json)

        # 为JSON样本构建帧路径列表 (input_depth个帧)
        half_depth_test = test_input_depth // 2
        for offset_test in range(-half_depth_test, half_depth_test + (test_input_depth % 2)): # 确保总共 test_input_depth 帧
            target_frame_actual_idx = center_frame_actual_idx + offset_test
            # 如果超出范围，则填充边界帧
            actual_idx_to_load = max(0, min(target_frame_actual_idx, total_dummy_frames_in_raw - 1))
            frame_load_basename = f"{seq_name_test}_frame{actual_idx_to_load:03d}.png"
            frame_paths_for_json_sample.append(str(Path(seq_name_test) / frame_load_basename))
        
        if len(frame_paths_for_json_sample) == test_input_depth:
             dummy_samples_data_for_json.append({
                 "sample_id": f"{seq_name_test}_center{center_frame_actual_idx}",
                 "frames": frame_paths_for_json_sample,
                 "mask_file": mask_path_for_json_sample
             })
    
    if not dummy_samples_data_for_json:
        print("未能为JSON创建任何伪样本。请检查 INPUT_DEPTH 和伪数据逻辑。")

    dummy_json_content_for_test = {
        "dataset_name": "Dummy_Debug_SemDataset",
        "root_raw_image_dir": raw_dir_name_for_test, # 使用配置中的目录名
        "root_labeled_mask_dir": labeled_dir_name_for_test, # 使用配置中的目录名
        "input_depth": test_input_depth,
        "num_samples": len(dummy_samples_data_for_json),
        "samples": dummy_samples_data_for_json
    }
    with open(dummy_json_path, 'w') as f:
        json.dump(dummy_json_content_for_test, f, indent=4)
    print(f"创建了用于测试的伪JSON: {dummy_json_path}")


    # --- 3. 定义示例变换 (使用 augmentations_utils) ---
    # 期望的图像尺寸 (可以与伪图像的实际保存尺寸不同，增强会处理)
    target_h, target_w = test_img_h, test_img_w 

    # 从 augmentations_utils 获取训练增强
    # 注意：这里的 target_h, target_w 应该是模型期望的输入尺寸
    example_augmentations = augmentations_utils.get_train_augmentations(
        height=target_h, 
        width=target_w
    )
    print(f"获取的训练增强: {example_augmentations is not None}")

    # 从 augmentations_utils 获取归一化变换
    # 为了测试，这里可以不从文件加载，或者确保有一个假的统计文件
    # (在 augmentations_utils 的测试部分已经创建了 dummy_test_stats.json)
    # 我们假设 augmentations_utils.py 的测试部分已在 json_dir_abs_for_test 创建了一个有效的统计文件
    # 或者，我们可以直接传递均值/标准差
    # mean_test, std_test = [0.5]*test_input_depth, [0.5]*test_input_depth
    # example_normalization = augmentations_utils.get_normalization_transform(mean=mean_test, std=std_test)

    # 尝试从文件加载（确保 augmentations_utils 的测试部分已运行或手动创建文件）
    temp_stats_file_for_dataset_test = json_dir_abs_for_test / "temp_dataset_stats_for_sem_test.json"
    if test_input_depth > 0:
        with open(temp_stats_file_for_dataset_test, 'w') as f_stats:
            json.dump({
                "mean": [0.45] * test_input_depth, "std": [0.25] * test_input_depth,
                "input_depth_at_calculation": test_input_depth
            }, f_stats)
        
        example_normalization = augmentations_utils.get_normalization_transform(
            load_from_file=True,
            stats_json_name=temp_stats_file_for_dataset_test.name, # 使用刚创建的临时文件
            project_root=current_project_root_for_test,
            json_dir_name_relative_to_project=json_dir_name_for_test,
            input_depth_for_stats_check=test_input_depth
        )
        print(f"获取的归一化变换: {example_normalization is not None}")
    else:
        example_normalization = None
        print("test_input_depth 为0，跳过归一化变换的获取。")


    # --- 4. 实例化数据集 ---
    print(f"\n使用 {dummy_json_filename} 实例化 SemSegmentationDataset...")
    try:
        test_dataset = SemSegmentationDataset(
            json_file_identifier=str(dummy_json_filename),
            project_root=current_project_root_for_test,
            input_depth_from_config=test_input_depth,
            class_mapping_from_config=test_class_mapping,
            json_dir_name_relative_to_project=json_dir_name_for_test,
            augmentations=example_augmentations,
            normalization_transform=example_normalization
        )

        if len(test_dataset) > 0:
            print(f"成功创建数据集，包含 {len(test_dataset)} 个样本。")

            # --- 5. 从数据集中获取一个样本 ---
            print("\n从数据集中获取一个样本...")
            try:
                image_tensor, mask_tensor = test_dataset[0]
                print(f"  样本 0 - 图像张量形状: {image_tensor.shape}, 数据类型: {image_tensor.dtype}")
                print(f"  样本 0 - 掩码张量形状: {mask_tensor.shape}, 数据类型: {mask_tensor.dtype}")
                if mask_tensor.numel() > 0:
                    print(f"  样本 0 - 掩码值 最小值: {mask_tensor.min()}, 最大值: {mask_tensor.max()}")
                    if num_classes_test > 0 and mask_tensor.max() >= num_classes_test:
                         print(f"  警告: 最大掩码值 {mask_tensor.max()} >= NUM_CLASSES {num_classes_test}。")
            except Exception as e:
                print(f"从数据集中获取样本时出错: {e}")
                import traceback
                traceback.print_exc()

            # --- 6. 使用 DataLoader 测试 ---
            if len(test_dataset) > 0 :
                print("\n使用 DataLoader 测试...")
                try:
                    batch_s = min(2, len(test_dataset)) if len(test_dataset) > 0 else 1
                    if batch_s == 0 and len(test_dataset) > 0: batch_s = 1
                    
                    if batch_s > 0 :
                        # num_workers > 0 在Windows上可能需要在 if __name__ == '__main__' 块中
                        # 对于测试，设为0更简单
                        test_loader = DataLoader(test_dataset, batch_size=batch_s, shuffle=True, num_workers=0)
                        for i, (images, masks) in enumerate(test_loader):
                            print(f"  批次 {i+1}:")
                            print(f"    图像批次形状: {images.shape}, 数据类型: {images.dtype}")
                            print(f"    掩码批次形状: {masks.shape}, 数据类型: {masks.dtype}")
                            if i >= 0: # 测试一个批次就足够了
                                break
                        print("DataLoader 测试成功 (一个批次)。")
                    else:
                        print("无法创建 DataLoader，数据集在伪创建后似乎长度为0。")
                except Exception as e:
                    print(f"DataLoader 迭代过程中出错: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            print("测试数据集为空。跳过样本获取和 DataLoader 测试。")

    except FileNotFoundError as e:
        print(f"数据集实例化过程中出错 (FileNotFound): {e}")
    except ValueError as e:
        print(f"数据集实例化过程中出错 (ValueError): {e}")
    except Exception as e:
        print(f"数据集测试过程中发生意外错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理临时统计文件
        if temp_stats_file_for_dataset_test.exists():
            temp_stats_file_for_dataset_test.unlink()

    print("\n--- SemSegmentationDataset 测试运行结束 ---")