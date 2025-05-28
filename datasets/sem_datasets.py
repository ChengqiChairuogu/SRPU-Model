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
from configs import base as cfg_base
from configs import json_config as cfg_json_gen

class SemSegmentationDataset(Dataset):
    def __init__(self,
                 json_file_identifier: str, 
                 project_root: Path,
                 input_depth_from_config: int, 
                 class_mapping_from_config: Dict[Tuple[int, int, int], int],
                 json_dir_name_relative_to_project: str,
                 image_transform: Optional[Callable] = None,
                 augmentations: Optional[Callable] = None):

        self.project_root = project_root
        self.input_depth = input_depth_from_config
        self.class_mapping = class_mapping_from_config
        self.image_transform = image_transform
        self.augmentations = augmentations

        json_file_path_obj = Path(json_file_identifier)
        if json_file_path_obj.is_absolute():
            self.json_file_abs_path = json_file_path_obj
        else:
            # 先检查 json_file_identifier 是否已经是 "json/train.json" 这种形式
            potential_path_as_is = self.project_root / json_file_identifier
            if potential_path_as_is.exists() and json_dir_name_relative_to_project in json_file_identifier:
                 self.json_file_abs_path = potential_path_as_is
            else: # 如果 json_file_identifier 只是文件名 "train.json"
                 self.json_file_abs_path = self.project_root / json_dir_name_relative_to_project / json_file_identifier
        
        if not self.json_file_abs_path.exists():
            # 尝试另一种可能：如果json_file_identifier只是文件名，并且已经位于配置的json目录下
            alt_path = self.project_root / json_dir_name_relative_to_project / Path(json_file_identifier).name
            if alt_path.exists():
                self.json_file_abs_path = alt_path
            else:
                raise FileNotFoundError(f"Resolved JSON file not found. Checked: {self.json_file_abs_path} and {alt_path}")

        try:
            with open(self.json_file_abs_path, 'r') as f:
                self.data_info = json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"Error decoding JSON from: {self.json_file_abs_path}")

        self.samples = self.data_info.get("samples", [])
        if not self.samples:
            print(f"Warning: No samples found in {self.json_file_abs_path}.")

        self.raw_image_root_name_in_json = self.data_info.get("root_raw_image_dir")
        self.mask_root_name_in_json = self.data_info.get("root_labeled_mask_dir") 

        if not self.raw_image_root_name_in_json:
            raise ValueError(f"'root_raw_image_dir' not found in JSON file: {self.json_file_abs_path}")

        json_input_depth = self.data_info.get("input_depth")
        if json_input_depth is not None and json_input_depth != self.input_depth:
            print(f"Warning: input_depth from JSON ({json_input_depth}) differs from "
                  f"config/parameter ({self.input_depth}). Using {self.input_depth} from parameters.")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image_frame(self, frame_relative_path: str) -> np.ndarray:

        image_path = self.project_root / self.raw_image_root_name_in_json / frame_relative_path
        try:
            img = Image.open(image_path)
            if img.mode not in ['L', 'RGB', 'RGBA', 'I', 'F']:
                 img = img.convert('L') 
            elif img.mode == 'RGBA':
                 img = img.convert('RGB')
            return np.array(img)
        except FileNotFoundError:
            raise FileNotFoundError(f"Image frame not found: {image_path}")
        except Exception as e:
            raise IOError(f"Error opening or processing image {image_path}: {e}")


    def _load_and_convert_mask(self, mask_relative_path: str) -> Optional[np.ndarray]:
        if not mask_relative_path or not self.mask_root_name_in_json:
            return None

        mask_path = self.project_root / self.mask_root_name_in_json / mask_relative_path
        try:
            mask_pil = Image.open(mask_path)
        except FileNotFoundError:
            return None 

        mask_np_original = np.array(mask_pil)

        if mask_np_original.ndim == 2:
            return mask_np_original.astype(np.uint8)
        elif mask_np_original.ndim == 3 and self.class_mapping:
            if mask_np_original.shape[-1] == 4: 
                mask_np_original = mask_np_original[..., :3]
            if mask_np_original.shape[-1] != 3:
                raise ValueError(f"Mask {mask_path} is 3-dimensional but not RGB (shape: {mask_np_original.shape}). Cannot apply RGB class_mapping.")

            mask_np_labels = np.zeros((mask_np_original.shape[0], mask_np_original.shape[1]), dtype=np.uint8)
            for color_rgb_tuple, class_idx in self.class_mapping.items():
                match = np.all(mask_np_original == np.array(color_rgb_tuple, dtype=np.uint8), axis=-1)
                mask_np_labels[match] = class_idx
            return mask_np_labels
        elif mask_np_original.ndim == 3 and not self.class_mapping:
            return mask_np_original[..., 0].astype(np.uint8)
        else:
            raise ValueError(f"Unsupported mask format or dimension: {mask_np_original.shape} for mask {mask_path}")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.samples: 
            raise IndexError("Dataset is empty or samples were not loaded correctly.")
        if idx >= len(self.samples):
            raise IndexError("Index out of bounds")
            
        sample_info = self.samples[idx]
        
        image_frame_paths_in_json: List[str] = sample_info["frames"] 
        mask_file_path_in_json: Optional[str] = sample_info.get("mask_file") 

        # 1. 加载并堆叠图像帧
        img_frames_list_np: List[np.ndarray] = []
        for frame_relative_path in image_frame_paths_in_json:
            img_np = self._load_image_frame(frame_relative_path) 
            if img_np.ndim == 2: 
                img_np = np.expand_dims(img_np, axis=-1) 
            elif img_np.ndim == 3 and img_np.shape[-1] != 1 and img_np.shape[-1] != 3 :
                if img_np.shape[-1] > 1 :
                     img_np = img_np[..., [0]]
            img_frames_list_np.append(img_np)
        
        try:
            stacked_images_np = np.concatenate(img_frames_list_np, axis=-1).astype(np.float32)
        except ValueError as e:
            print(f"Error concatenating image frames for sample_id: {sample_info.get('sample_id', 'N/A')}")
            print(f"Paths: {image_frame_paths_in_json}")
            print(f"Shapes of loaded frames before concat: {[img.shape for img in img_frames_list_np]}")
            raise e

        if stacked_images_np.shape[-1] != self.input_depth:
            pass

        # 2. 加载并处理掩码
        mask_list_np: List[np.ndarray] = []
        center_mask_np: Optional[np.ndarray] = None
        h_ref_for_dummy_mask, w_ref_for_dummy_mask = stacked_images_np.shape[:2]

        if mask_file_path_in_json: 
            center_mask_np = self._load_and_convert_mask(mask_file_path_in_json)
            if center_mask_np is None:
                 center_mask_np = np.zeros((h_ref_for_dummy_mask, w_ref_for_dummy_mask), dtype=np.uint8)
        
        for i in range(self.input_depth): 
            if center_mask_np is not None:
                mask_list_np.append(center_mask_np.copy()) 
            else: 
                mask_list_np.append(np.zeros((h_ref_for_dummy_mask, w_ref_for_dummy_mask), dtype=np.uint8))
        
        stacked_masks_np = np.stack(mask_list_np, axis=0).astype(np.int64) 

        # 3. 应用数据增强 (Albumentations)
        if self.augmentations:
            try:
                if stacked_masks_np.shape[0] == 1 and not isinstance(self.augmentations, list): # Single mask channel
                    augmented = self.augmentations(image=stacked_images_np, mask=stacked_masks_np[0])
                    stacked_images_np = augmented['image']
                    stacked_masks_np = np.expand_dims(augmented['mask'], axis=0) # Add channel dim back
                elif stacked_masks_np.shape[0] > 1 and not isinstance(self.augmentations, list): # Multiple mask channels
                    masks_for_aug = [m for m in stacked_masks_np]
                    augmented = self.augmentations(image=stacked_images_np, masks=masks_for_aug)
                    stacked_images_np = augmented['image']
                    stacked_masks_np = np.stack(augmented['masks'], axis=0)
                else: # Fallback or if augmentations handles (D,H,W) directly
                     augmented = self.augmentations(image=stacked_images_np, mask=stacked_masks_np)
                     stacked_images_np = augmented['image']
                     stacked_masks_np = augmented['mask']

            except Exception as e: 
                print(f"Error during augmentation for sample {sample_info.get('sample_id', idx)}: {e}")

        # 4. 应用图像转换 (ToTensor, Normalize)
        image_tensor = torch.from_numpy(stacked_images_np.transpose((2, 0, 1))).float()
        if self.image_transform: 
            image_tensor = self.image_transform(image_tensor)

        mask_tensor = torch.from_numpy(stacked_masks_np).long() 
        
        return image_tensor, mask_tensor




# ======================================================================
# Test / Debug block
# ======================================================================
if __name__ == '__main__':
    print("--- Running SemSegmentationDataset directly for testing ---")

    # --- 1. Use (or create minimal) Configs ---
    current_project_root_for_test = cfg_base.PROJECT_ROOT.resolve()
    json_dir_for_test = current_project_root_for_test / cfg_json_gen.JSON_OUTPUT_DIR_NAME
    raw_dir_for_test = current_project_root_for_test / cfg_json_gen.RAW_IMAGE_SOURCE_DIR_NAME
    labeled_dir_for_test = current_project_root_for_test / cfg_json_gen.LABELED_MASK_DIR_NAME

    json_dir_for_test.mkdir(parents=True, exist_ok=True)
    raw_dir_for_test.mkdir(parents=True, exist_ok=True)
    labeled_dir_for_test.mkdir(parents=True, exist_ok=True)


    # --- 2. Create a Dummy JSON file for testing ---
    dummy_json_filename = "test_debug_dataset.json"
    dummy_json_path = json_dir_for_test / dummy_json_filename
    
    num_dummy_frames_per_sequence = cfg_json_gen.INPUT_DEPTH 
    dummy_img_h = getattr(cfg_base, 'IMAGE_HEIGHT', 64) // 2 
    dummy_img_w = getattr(cfg_base, 'IMAGE_WIDTH', 64) // 2

    dummy_samples_data_for_json = []

    seq_name_test = "debug_seq1"
    raw_seq_dir = raw_dir_for_test / seq_name_test
    labeled_seq_dir = labeled_dir_for_test / seq_name_test
    raw_seq_dir.mkdir(parents=True, exist_ok=True)
    labeled_seq_dir.mkdir(parents=True, exist_ok=True)

    frame_paths_for_json_sample = []
    mask_path_for_json_sample = None

    total_dummy_frames_in_raw = num_dummy_frames_per_sequence + 2 
    for frame_idx in range(total_dummy_frames_in_raw):
        frame_basename = f"{seq_name_test}_frame{frame_idx:03d}.png"
        dummy_image_rel_path = Path(seq_name_test) / frame_basename
        dummy_image_abs_path = raw_dir_for_test / dummy_image_rel_path
        if not dummy_image_abs_path.exists():
            Image.new('L', (dummy_img_w, dummy_img_h), color='gray').save(dummy_image_abs_path)

    center_frame_actual_idx = 1 # This is the index in the raw sequence (0, 1, 2, 3, 4...)
    if center_frame_actual_idx < total_dummy_frames_in_raw:
        # Create mask for this center frame
        center_frame_basename = f"{seq_name_test}_frame{center_frame_actual_idx:03d}.png"
        dummy_mask_rel_path = Path(seq_name_test) / center_frame_basename # Mask has same name as image
        dummy_mask_abs_path = labeled_dir_for_test / dummy_mask_rel_path
        if not dummy_mask_abs_path.exists():
             mask_array = np.zeros((dummy_img_h, dummy_img_w), dtype=np.uint8)
             mask_array[dummy_img_h//4:dummy_img_h//2, dummy_img_w//4:dummy_img_w//2] = 1 
             Image.fromarray(mask_array, mode='L').save(dummy_mask_abs_path)
        mask_path_for_json_sample = str(dummy_mask_rel_path)

        half_depth_test = cfg_json_gen.INPUT_DEPTH // 2
        for offset_test in range(-half_depth_test, half_depth_test + 1):
            target_frame_actual_idx = center_frame_actual_idx + offset_test
            # Pad with boundary frames if needed
            actual_idx_to_load = max(0, min(target_frame_actual_idx, total_dummy_frames_in_raw - 1))
            frame_load_basename = f"{seq_name_test}_frame{actual_idx_to_load:03d}.png"
            frame_paths_for_json_sample.append(str(Path(seq_name_test) / frame_load_basename))
        
        if len(frame_paths_for_json_sample) == cfg_json_gen.INPUT_DEPTH:
             dummy_samples_data_for_json.append({
                 "sample_id": f"{seq_name_test}_center{center_frame_actual_idx}",
                 "frames": frame_paths_for_json_sample,
                 "mask_file": mask_path_for_json_sample
             })
    
    if not dummy_samples_data_for_json:
        print("Failed to create any dummy samples for JSON. Check INPUT_DEPTH and dummy data logic.")

    dummy_json_content_for_test = {
        "dataset_name": "Dummy_Debug_Dataset",
        "root_raw_image_dir": cfg_json_gen.RAW_IMAGE_SOURCE_DIR_NAME,
        "root_labeled_mask_dir": cfg_json_gen.LABELED_MASK_DIR_NAME,
        "input_depth": cfg_json_gen.INPUT_DEPTH,
        "num_samples": len(dummy_samples_data_for_json),
        "samples": dummy_samples_data_for_json
    }
    with open(dummy_json_path, 'w') as f:
        json.dump(dummy_json_content_for_test, f, indent=4)
    print(f"Created dummy JSON for testing: {dummy_json_path}")


    # --- 3. Define Example Transforms ---
    example_image_transform = None
    if hasattr(cfg_base, 'IMG_MEAN') and hasattr(cfg_base, 'IMG_STD'):
        example_image_transform = transforms.Compose([
            transforms.Normalize(mean=cfg_base.IMG_MEAN, std=cfg_base.IMG_STD)
        ])
    else:
        print("cfg_base.IMG_MEAN or cfg_base.IMG_STD not found. No normalization transform will be applied.")
    
    example_augmentations = None # Keep it simple for basic test


    # --- 4. Instantiate the Dataset ---
    print(f"\nInstantiating SemSegmentationDataset with {dummy_json_filename}...")
    try:
        test_dataset = SemSegmentationDataset(
            json_file_identifier=str(dummy_json_filename), 
            project_root=current_project_root_for_test, # Use the one from cfg_base
            input_depth_from_config=cfg_json_gen.INPUT_DEPTH,
            class_mapping_from_config=cfg_base.MAPPING,
            json_dir_name_relative_to_project=cfg_json_gen.JSON_OUTPUT_DIR_NAME,
            image_transform=example_image_transform,
            augmentations=example_augmentations
        )

        if len(test_dataset) > 0:
            print(f"Successfully created dataset with {len(test_dataset)} samples.")


            # --- 5. Get a sample ---
            print("\nFetching one sample from the dataset...")
            try:
                image_tensor, mask_tensor = test_dataset[0]
                print(f"  Sample 0 - Image Tensor Shape: {image_tensor.shape}, Dtype: {image_tensor.dtype}")
                print(f"  Sample 0 - Mask Tensor Shape: {mask_tensor.shape}, Dtype: {mask_tensor.dtype}")
                if mask_tensor.numel() > 0:
                    print(f"  Sample 0 - Mask values min: {mask_tensor.min()}, max: {mask_tensor.max()}")
                    if hasattr(cfg_base, 'NUM_CLASSES') and mask_tensor.max() >= cfg_base.NUM_CLASSES:
                         print(f"  Warning: Max mask value {mask_tensor.max()} is >= NUM_CLASSES {cfg_base.NUM_CLASSES}.")
            except Exception as e:
                print(f"Error getting sample from dataset: {e}")
                import traceback
                traceback.print_exc()


            # --- 6. Test with DataLoader ---
            if len(test_dataset) > 0 :
                print("\nTesting with DataLoader...")
                try:
                    # Use batch_size=1 if only one sample was created by dummy data logic
                    batch_s = min(2, len(test_dataset)) if len(test_dataset) > 0 else 1
                    if batch_s == 0 and len(test_dataset) > 0: batch_s = 1 # Ensure batch_size is at least 1 if dataset not empty

                    if batch_s > 0 :
                        test_loader = DataLoader(test_dataset, batch_size=batch_s, shuffle=True)
                        for i, (images, masks) in enumerate(test_loader):
                            print(f"  Batch {i+1}:")
                            print(f"    Images batch shape: {images.shape}, dtype: {images.dtype}")
                            print(f"    Masks batch shape: {masks.shape}, dtype: {masks.dtype}")
                            if i >= 0: 
                                break
                        print("DataLoader test successful for one batch.")
                    else:
                        print("Cannot create DataLoader, dataset seems to have 0 length after dummy creation.")
                except Exception as e:
                    print(f"Error during DataLoader iteration: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            print("Test dataset is empty. Skipping sample fetching and DataLoader test.")

    except FileNotFoundError as e:
        print(f"Error during dataset instantiation (FileNotFound): {e}")
    except ValueError as e:
        print(f"Error during dataset instantiation (ValueError): {e}")
    except Exception as e:
        print(f"An unexpected error occurred during dataset testing: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Finished SemSegmentationDataset test run ---")