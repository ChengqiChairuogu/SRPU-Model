# tasks/enhanced_active_learning_task.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path
import sys
import os
import json
from tqdm import tqdm
import shutil
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from typing import Tuple

# 添加项目根目录到sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from datasets.sem_datasets import SemSegmentationDataset
from models.segmentation_unet import SegmentationUNet
from utils.training_util import get_loss_function, train_one_epoch, evaluate_model
from utils.logging_util import Logger
from utils.augmentation_util import load_dataset_stats
from utils.uncertainty_util import compute_combined_uncertainty, detect_prediction_boundaries
from configs.active_learning import active_learning_config as cfg_al
from configs import base as cfg_base

class EnhancedActiveLearningTask:
    """增强版主动学习任务，集成SAM提示词功能"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.sam_tool = None
        self.logger = None
        
    def initialize_sam(self, sam_checkpoint_path: str = None):
        """初始化SAM工具"""
        try:
            from utils.sam_annotation_util import SAMAnnotationTool
            self.sam_tool = SAMAnnotationTool(sam_checkpoint_path)
            print("SAM工具初始化成功")
        except Exception as e:
            print(f"SAM工具初始化失败: {e}")
            self.sam_tool = None
    
    def create_model(self, encoder_name: str, decoder_name: str) -> nn.Module:
        """创建分割模型"""
        print(f"创建模型: Encoder: {encoder_name}, Decoder: {decoder_name}")
        
        if encoder_name == 'unet':
            from models.encoders.unet_encoder import UNetEncoder
            encoder = UNetEncoder(n_channels=cfg_base.INPUT_DEPTH)
        else:
            raise ValueError(f"未知的编码器: {encoder_name}")
        
        encoder_channels = encoder.get_channels()
        if decoder_name == 'unet':
            from models.decoders.unet_decoder import UNetDecoder
            decoder = UNetDecoder(encoder_channels, n_classes=cfg_base.NUM_CLASSES)
        else:
            raise ValueError(f"未知的解码器: {decoder_name}")
        
        return SegmentationUNet(encoder, decoder)
    
    def load_pretrained_model(self, model_path: str):
        """加载预训练模型"""
        self.model = self.create_model(cfg_al.ENCODER_NAME, cfg_al.DECODER_NAME).to(self.device)
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"预训练模型已加载: {model_path}")
        else:
            print(f"预训练模型不存在: {model_path}")
    
    def predict_with_uncertainty(self, image_path: str) -> Tuple[np.ndarray, float]:
        """预测图像并计算不确定性"""
        # 加载和预处理图像
        image_pil = Image.open(image_path).convert('L')
        image_np = np.array(image_pil, dtype=np.float32) / 255.0
        image_stack = np.stack([image_np] * cfg_base.INPUT_DEPTH, axis=-1)
        
        # 数据增强
        transform = self._build_transforms()
        augmented = transform(image=image_stack)
        image_tensor = augmented['image'].unsqueeze(0).to(self.device)
        
        # 模型预测
        self.model.eval()
        with torch.no_grad():
            logits = self.model(image_tensor)
            pred_mask = torch.argmax(logits, dim=1)
            
            # 计算不确定性
            uncertainty = compute_combined_uncertainty(
                logits, 
                cfg_al.UNCERTAINTY_METHODS,
                prediction_mask=pred_mask,
                dilation_size=cfg_al.BOUNDARY_DILATION_SIZE
            )
            
            # 调整到原始尺寸
            original_size = image_pil.size[::-1]
            logits_resized = F.interpolate(logits, size=original_size, mode='bilinear', align_corners=False)
            pred_mask_resized = torch.argmax(logits_resized, dim=1).squeeze(0).cpu().numpy()
            
            return pred_mask_resized, uncertainty.item()
    
    def _build_transforms(self):
        """构建数据预处理流水线"""
        mean, std = load_dataset_stats(expected_input_depth=cfg_base.INPUT_DEPTH)
        
        if not mean or not std:
            return A.Compose([
                A.Resize(height=cfg_base.IMAGE_HEIGHT, width=cfg_base.IMAGE_WIDTH),
                ToTensorV2(),
            ])
        
        return A.Compose([
            A.Resize(height=cfg_base.IMAGE_HEIGHT, width=cfg_base.IMAGE_WIDTH),
            A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
            ToTensorV2(),
        ])
    
    def generate_sam_correction_interface(self, image_path: str, prediction_mask: np.ndarray):
        """生成SAM修正界面"""
        if not self.sam_tool:
            print("SAM工具未初始化")
            return
        
        # 加载图像到SAM
        self.sam_tool.load_image(image_path)
        
        # 创建修正界面
        self._create_correction_interface(image_path, prediction_mask)
    
    def apply_sam_correction_with_points(self, image_path: str, points: list, labels: list) -> np.ndarray:
        """使用点击点进行SAM修正"""
        if not self.sam_tool:
            print("SAM工具未初始化")
            return None
        
        self.sam_tool.load_image(image_path)
        sam_mask = self.sam_tool.segment_with_points(points, labels)
        return sam_mask
    
    def apply_sam_correction_with_box(self, image_path: str, box: list) -> np.ndarray:
        """使用边界框进行SAM修正"""
        if not self.sam_tool:
            print("SAM工具未初始化")
            return None
        
        self.sam_tool.load_image(image_path)
        sam_mask = self.sam_tool.segment_with_box(box)
        return sam_mask
    
    def _create_correction_interface(self, image_path: str, prediction_mask: np.ndarray):
        """创建修正界面"""
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, RadioButtons
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 显示原始图像
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax1.imshow(image_rgb)
        ax1.set_title("原始图像")
        ax1.axis('off')
        
        # 显示预测mask
        ax2.imshow(prediction_mask)
        ax2.set_title("预测Mask")
        ax2.axis('off')
        
        # 添加控制按钮
        ax_button = plt.axes([0.1, 0.05, 0.8, 0.1])
        button_sam = Button(ax_button, '使用SAM修正')
        
        def on_sam_click(event):
            self._apply_sam_correction(image_path, prediction_mask)
        
        button_sam.on_clicked(on_sam_click)
        
        plt.tight_layout()
        plt.show()
    
    def _apply_sam_correction(self, image_path: str, prediction_mask: np.ndarray):
        """应用SAM修正"""
        if not self.sam_tool:
            print("SAM工具未初始化")
            return
        
        print("=== SAM智能修正 ===")
        print("1. 点击图像中的前景区域")
        print("2. 使用边界框选择区域")
        print("3. 结合文本描述进行分割")
        
        # 加载图像到SAM
        self.sam_tool.load_image(image_path)
        
        # 示例：使用点击分割
        # 这里可以添加交互式点击功能
        print("示例：使用点击分割")
        points = [(100, 100), (200, 200)]  # 示例点击点
        labels = [1, 1]  # 前景点
        
        sam_mask = self.sam_tool.segment_with_points(points, labels)
        if sam_mask is not None:
            print("SAM分割成功")
            # 这里可以将SAM结果与预测mask结合
            return sam_mask
        else:
            print("SAM分割失败")
            return prediction_mask
    
    def run_enhanced_active_learning(self):
        """运行增强版主动学习"""
        print("=== 增强版主动学习任务 ===")
        
        # 初始化SAM
        self.initialize_sam()
        
        # 加载模型
        self.load_pretrained_model(cfg_al.PRETRAINED_MODEL_PATH)
        
        # 创建目录
        (project_root / cfg_al.PREDICTIONS_DIR).mkdir(parents=True, exist_ok=True)
        (project_root / cfg_al.SELECTION_INFO_DIR).mkdir(parents=True, exist_ok=True)
        
        # 初始化日志
        self.logger = Logger(cfg_al.log_config)
        
        for iter_num in range(1, cfg_al.NUM_ITERATIONS + 1):
            print(f"\n=== 主动学习迭代 {iter_num} ===")
            
            # 1. 选择样本
            selected_image = self._select_sample()
            if not selected_image:
                print("没有找到合适的样本")
                continue
            
            # 2. 生成预测
            prediction_mask, uncertainty = self.predict_with_uncertainty(selected_image)
            
            # 3. 保存预测结果
            self._save_prediction_results(iter_num, selected_image, prediction_mask, uncertainty)
            
            # 4. 提供SAM修正界面
            if self.sam_tool:
                self.generate_sam_correction_interface(selected_image, prediction_mask)
            
            # 5. 等待用户标注
            input("完成标注后按Enter继续...")
            
            # 6. 模型微调
            self._fine_tune_model()
        
        print("增强版主动学习任务完成")
    
    def _select_sample(self) -> str:
        """选择样本"""
        unlabeled_dir = project_root / cfg_al.UNLABELED_POOL_DIR
        image_files = [f for f in os.listdir(unlabeled_dir) if f.endswith('.png')]
        
        if not image_files:
            return None
        
        # 计算所有图像的不确定性
        uncertainties = []
        for image_file in tqdm(image_files, desc="计算不确定性"):
            image_path = unlabeled_dir / image_file
            try:
                _, uncertainty = self.predict_with_uncertainty(str(image_path))
                uncertainties.append((image_file, uncertainty))
            except Exception as e:
                print(f"处理图像 {image_file} 时出错: {e}")
                continue
        
        # 选择不确定性最高的样本
        if uncertainties:
            selected = max(uncertainties, key=lambda x: x[1])
            return str(unlabeled_dir / selected[0])
        
        return None
    
    def _save_prediction_results(self, iter_num: int, image_path: str, 
                               prediction_mask: np.ndarray, uncertainty: float):
        """保存预测结果"""
        # 创建迭代目录
        iteration_dir = project_root / cfg_al.PREDICTIONS_DIR / f"iteration_{iter_num}"
        iteration_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存预测mask
        rgb_mask = self._class_to_rgb(prediction_mask)
        mask_filename = f"{Path(image_path).stem}_prediction_mask.png"
        mask_path = iteration_dir / mask_filename
        Image.fromarray(rgb_mask).save(mask_path)
        
        # 复制原始图像
        original_copy_path = iteration_dir / Path(image_path).name
        shutil.copy2(image_path, original_copy_path)
        
        # 保存选择信息
        selection_info = {
            "image_path": str(image_path),
            "uncertainty": uncertainty,
            "prediction_mask_path": str(mask_path),
            "iteration": iter_num
        }
        
        info_path = project_root / cfg_al.SELECTION_INFO_DIR / f"iteration_{iter_num}_info.json"
        with open(info_path, 'w') as f:
            json.dump(selection_info, f, indent=4)
        
        print(f"预测结果已保存到: {iteration_dir}")
        print(f"不确定性分数: {uncertainty:.4f}")
    
    def _class_to_rgb(self, pred_mask: np.ndarray) -> np.ndarray:
        """将类别掩码转换为RGB图像"""
        h, w = pred_mask.shape
        rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_idx, color in cfg_base.COLOR_MAPPING.items():
            rgb_mask[pred_mask == class_idx] = color
        
        return rgb_mask
    
    def _fine_tune_model(self):
        """模型微调"""
        print("开始模型微调...")
        
        # 加载训练数据
        train_dataset = SemSegmentationDataset(
            json_file_identifier=cfg_al.TRAIN_JSON_NAME,
            project_root=project_root,
            split='train'
        )
        train_loader = DataLoader(train_dataset, batch_size=cfg_al.BATCH_SIZE, shuffle=True)
        
        # 训练设置
        criterion = get_loss_function(cfg_al.LOSS_FUNCTION)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg_al.LEARNING_RATE)
        
        # 微调训练
        for epoch in range(cfg_al.NUM_EPOCHS_PER_ITER):
            train_loss = train_one_epoch(self.model, train_loader, optimizer, criterion, self.device)
            print(f"Epoch {epoch+1}, Loss: {train_loss:.4f}")
        
        print("模型微调完成")


def main_enhanced_active_learning():
    """主函数"""
    task = EnhancedActiveLearningTask()
    task.run_enhanced_active_learning()


if __name__ == '__main__':
    main_enhanced_active_learning() 