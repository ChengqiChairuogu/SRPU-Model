# utils/sam_annotation_util.py
import torch
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

try:
    from segment_anything import SamPredictor, sam_model_registry
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("警告: SAM未安装，请运行: pip install segment-anything")

class SAMAnnotationTool:
    """SAM集成的智能标注工具"""
    
    def __init__(self, sam_checkpoint_path: str = None, model_type: str = "vit_h"):
        self.sam_predictor = None
        self.current_image = None
        self.current_mask = None
        
        if SAM_AVAILABLE and sam_checkpoint_path:
            self._initialize_sam(sam_checkpoint_path, model_type)
    
    def _initialize_sam(self, checkpoint_path: str, model_type: str):
        """初始化SAM模型"""
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            sam.to(device=device)
            self.sam_predictor = SamPredictor(sam)
            print(f"SAM模型已加载到设备: {device}")
        except Exception as e:
            print(f"SAM模型加载失败: {e}")
    
    def load_image(self, image_path: str):
        """加载图像"""
        self.current_image = cv2.imread(image_path)
        self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        
        if self.sam_predictor:
            self.sam_predictor.set_image(self.current_image)
    
    def segment_with_points(self, points: List[Tuple[int, int]], 
                          labels: List[int]) -> np.ndarray:
        """使用点击点进行分割"""
        if not self.sam_predictor:
            return None
        
        try:
            masks, scores, _ = self.sam_predictor.predict(
                point_coords=np.array(points),
                point_labels=np.array(labels),
                multimask_output=True
            )
            return masks[np.argmax(scores)]
        except Exception as e:
            print(f"SAM预测失败: {e}")
            return None
    
    def segment_with_box(self, box: List[int]) -> np.ndarray:
        """使用边界框进行分割"""
        if not self.sam_predictor:
            return None
        
        try:
            masks, scores, _ = self.sam_predictor.predict(
                box=np.array(box),
                multimask_output=True
            )
            return masks[np.argmax(scores)]
        except Exception as e:
            print(f"SAM预测失败: {e}")
            return None 