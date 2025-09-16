"""
图像清晰度处理器
主要功能：
1. 计算每张图片的清晰度，生成分析图表和JSON文件
2. 对图像清晰度进行平均化处理，让所有图像的清晰度基本相等
3. 将处理后的图像保存到各自数据集目录下
4. 为新的数据集生成JSON文件用于训练
"""

import json
import logging
import sys
import os
from pathlib import Path, WindowsPath
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime

# 设置中文路径支持
if sys.platform == "win32":
    # Windows系统中文路径支持
    import locale
    try:
        # 尝试设置系统默认编码
        locale.setlocale(locale.LC_ALL, '')
    except:
        pass
    
    # 设置matplotlib中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

from configs.image_sharpness.sharpness_processor_config import *
from utils.image_sharpness_util import (
    SharpnessAssessor, 
    apply_sharpness_averaging,
    calculate_target_sharpness,
    find_reference_image,
    save_image_to_dataset
)
from utils.analysis_chart_util import SharpnessAnalysisChartGenerator


def safe_path_operation(func):
    """装饰器：安全处理中文路径"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (OSError, UnicodeEncodeError, UnicodeDecodeError) as e:
            print(f"路径操作错误: {e}")
            print(f"当前工作目录: {os.getcwd()}")
            print(f"Python版本: {sys.version}")
            print(f"平台: {sys.platform}")
            raise
    return wrapper


class ImageSharpnessProcessor:
    """图像清晰度处理器"""
    
    def __init__(self):
        """初始化处理器"""
        self.logger = self._setup_logging()
        
        # 初始化清晰度评估器
        self.assessor = SharpnessAssessor(methods=SHARPNESS_METHODS)
        
        # 初始化图表生成器
        self.chart_generator = SharpnessAnalysisChartGenerator()
        
        self.logger.info("图像清晰度处理器初始化完成")
    
    def _setup_logging(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _safe_path(self, path_str: str) -> Path:
        """安全处理中文路径"""
        try:
            # 尝试直接创建Path对象
            path = Path(path_str)
            return path
        except Exception as e:
            self.logger.warning(f"路径创建失败: {path_str}, 错误: {e}")
            try:
                # 尝试使用绝对路径
                if os.path.isabs(path_str):
                    path = Path(os.path.abspath(path_str))
                else:
                    path = Path(os.path.abspath(path_str))
                return path
            except Exception as e2:
                self.logger.error(f"绝对路径创建也失败: {path_str}, 错误: {e2}")
                # 最后尝试使用字符串路径
                return Path(str(path_str))
    
    def _safe_image_open(self, image_path: Path) -> np.ndarray:
        """安全打开图像文件"""
        try:
            # 尝试使用PIL打开
            with Image.open(str(image_path)) as img:
                return np.array(img)
        except Exception as e:
            self.logger.warning(f"PIL打开图像失败: {image_path}, 错误: {e}")
            try:
                # 尝试使用OpenCV打开
                img = cv2.imread(str(image_path))
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    raise ValueError(f"OpenCV无法读取图像: {image_path}")
            except Exception as e2:
                self.logger.error(f"OpenCV打开图像也失败: {image_path}, 错误: {e2}")
                raise
    
    def process_images(self) -> None:
        """处理图像的主流程"""
        self.logger.info("开始处理图像清晰度...")
        
        # 1. 评估原始图像清晰度
        self.logger.info("步骤1: 评估原始图像清晰度")
        original_sharpness = self._assess_original_images()
        
        # 2. 生成原始清晰度分析图表
        if GENERATE_BEFORE_CHARTS:
            self.logger.info("步骤2: 生成原始清晰度分析图表")
            self._generate_original_analysis_charts(original_sharpness)
        
        # 3. 保存清晰度数据到JSON
        self.logger.info("步骤3: 保存清晰度数据到JSON")
        self._save_sharpness_json(original_sharpness, "original")
        
        # 4. 应用清晰度平均化处理
        if ENABLE_SHARPNESS_AVERAGING:
            self.logger.info("步骤4: 应用清晰度平均化处理")
            self._apply_sharpness_averaging(original_sharpness)
        
        # 5. 评估处理后的图像清晰度
        self.logger.info("步骤5: 评估处理后的图像清晰度")
        processed_sharpness = self._assess_processed_images()
        
        # 6. 生成处理后的清晰度分析图表
        if GENERATE_AFTER_CHARTS:
            self.logger.info("步骤6: 生成处理后的清晰度分析图表")
            self._generate_processed_analysis_charts(original_sharpness, processed_sharpness)
        
        # 7. 保存处理后的清晰度数据到JSON
        self.logger.info("步骤7: 保存处理后的清晰度数据到JSON")
        self._save_sharpness_json(processed_sharpness, "processed")
        
        # 8. 生成训练用JSON文件
        if GENERATE_TRAINING_JSON or GENERATE_SSL_JSON:
            self.logger.info("步骤8: 生成训练用JSON文件")
            self._generate_training_jsons()
        
        # 9. 验证生成的JSON文件
        self.logger.info("步骤9: 验证生成的JSON文件")
        validation_success = self._validate_generated_jsons()
        
        if validation_success:
            self.logger.info("✅ 所有JSON文件验证通过！")
        else:
            self.logger.warning("⚠️ 部分JSON文件验证失败，请检查生成的文件")
        
        self.logger.info("图像清晰度处理完成！")
    
    def _assess_original_images(self) -> Dict[str, Dict[str, float]]:
        """评估原始图像清晰度"""
        if USE_DATASET_JSON and DATASET_JSON_PATH.exists():
            self.logger.info(f"使用数据集JSON文件模式: {DATASET_JSON_PATH}")
            self.logger.info("注意：此模式只处理有对应mask的图像")
            sharpness_data = self._assess_images_from_json()
        else:
            self.logger.info(f"使用文件夹扫描模式，处理所有原始图像")
            self.logger.info("此模式将处理所有数据集中的所有原始图像")
            sharpness_data = self._assess_images_from_folder()
        
        self.logger.info(f"成功评估 {len(sharpness_data)} 张图像")
        return sharpness_data
    
    def _assess_images_from_json(self) -> Dict[str, Dict[str, float]]:
        """从JSON文件读取图像列表并评估清晰度"""
        try:
            with open(DATASET_JSON_PATH, 'r', encoding='utf-8') as f:
                dataset_data = json.load(f)
            
            sharpness_data = {}
            total_samples = len(dataset_data.get("samples", []))
            
            for i, sample in enumerate(dataset_data.get("samples", [])):
                # 获取数据集名称和图像文件路径
                dataset_name = sample.get("dataset", "dataset1_LInCl")
                frames = sample.get("frames", [])
                
                for frame in frames:
                    # 根据数据集名称选择对应的输入文件夹
                    if dataset_name in DATASET_INPUT_PATHS:
                        img_path = DATASET_INPUT_PATHS[dataset_name] / frame
                    elif DATASET_IMAGE_FOLDER.exists():
                        img_path = DATASET_IMAGE_FOLDER / frame
                    else:
                        img_path = INPUT_FOLDER / frame
                    
                    if img_path.exists():
                        try:
                            # 评估图像清晰度
                            metrics = self.assessor.assess_image(img_path)
                            sharpness_data[str(img_path)] = metrics
                            
                            # 显示进度
                            if (i + 1) % 10 == 0:
                                self.logger.info(f"已处理 {i + 1}/{total_samples} 个样本")
                                
                        except Exception as e:
                            self.logger.warning(f"评估图像 {img_path} 失败: {e}")
                    else:
                        self.logger.warning(f"图像文件不存在: {img_path}")
            
            return sharpness_data
            
        except Exception as e:
            self.logger.error(f"读取数据集JSON文件失败: {e}")
            self.logger.info("回退到文件夹扫描模式")
            return self._assess_images_from_folder()
    
    def _assess_images_from_folder(self) -> Dict[str, Dict[str, float]]:
        """从文件夹扫描图像并评估清晰度"""
        sharpness_data = {}
        total_images = 0
        
        # 如果配置了多数据集输入路径，则扫描所有数据集
        try:
            from configs.image_sharpness.sharpness_processor_config import DATASET_INPUT_PATHS
            
            self.logger.info(f"开始扫描 {len(DATASET_INPUT_PATHS)} 个数据集...")
            
            for dataset_name, input_path in DATASET_INPUT_PATHS.items():
                if input_path.exists():
                    self.logger.info(f"扫描数据集 {dataset_name}: {input_path}")
                    
                    # 统计图像文件数量
                    image_files = list(input_path.glob("*.png"))
                    dataset_image_count = len(image_files)
                    total_images += dataset_image_count
                    
                    self.logger.info(f"  发现 {dataset_image_count} 张PNG图像")
                    
                    # 评估数据集中的图像
                    dataset_data = self.assessor.assess_batch(input_path)
                    sharpness_data.update(dataset_data)
                    
                    self.logger.info(f"  成功评估 {len(dataset_data)} 张图像")
                else:
                    self.logger.warning(f"数据集输入路径不存在: {input_path}")
            
            self.logger.info(f"文件夹扫描完成，总共处理 {total_images} 张图像")
            
        except ImportError:
            # 回退到原来的单文件夹扫描
            self.logger.info(f"回退到单文件夹扫描模式: {INPUT_FOLDER}")
            
            if not INPUT_FOLDER.exists():
                raise FileNotFoundError(f"输入文件夹不存在: {INPUT_FOLDER}")
            
            # 统计图像文件数量
            image_files = list(INPUT_FOLDER.glob("*.png"))
            total_images = len(image_files)
            self.logger.info(f"发现 {total_images} 张PNG图像")
            
            # 使用SharpnessAssessor评估图像
            sharpness_data = self.assessor.assess_batch(INPUT_FOLDER)
            self.logger.info(f"成功评估 {len(sharpness_data)} 张图像")
        
        return sharpness_data
    
    def _generate_original_analysis_charts(self, sharpness_data: Dict[str, Dict[str, float]]) -> None:
        """生成原始清晰度分析图表"""
        chart_dir = CHART_OUTPUT_FOLDER / "before_processing"
        chart_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成各种分析图表
        self.chart_generator.generate_sharpness_distribution(
            sharpness_data, 
            save_path=chart_dir / "sharpness_distribution.png"
        )
        
        self.chart_generator.generate_sharpness_comparison(
            sharpness_data,
            save_path=chart_dir / "sharpness_comparison.png"
        )
        
        self.chart_generator.generate_sharpness_statistics(
            sharpness_data,
            save_path=chart_dir / "sharpness_statistics.png"
        )
        
        self.logger.info(f"原始清晰度分析图表已保存到: {chart_dir}")
    
    def _save_sharpness_json(self, sharpness_data: Dict[str, Dict[str, float]], 
                            suffix: str) -> None:
        """保存清晰度数据到JSON文件"""
        # 准备JSON数据
        json_data = {
            "metadata": {
                "total_images": len(sharpness_data),
                "assessment_methods": SHARPNESS_METHODS,
                "timestamp": str(datetime.now()),
                "type": suffix
            },
            "images": {}
        }
        
        for img_path, metrics in sharpness_data.items():
            # 提取文件名
            img_name = Path(img_path).name
            json_data["images"][img_name] = {
                "file_path": str(img_path),
                "metrics": metrics
            }
        
        # 保存JSON文件
        json_path = JSON_OUTPUT_PATH.parent / f"sharpness_analysis_{suffix}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"清晰度数据已保存到: {json_path}")
    
    def _apply_sharpness_averaging(self, sharpness_data: Dict[str, Dict[str, float]]) -> None:
        """应用清晰度平均化处理"""
        # 计算目标清晰度
        target_sharpness = calculate_target_sharpness(
            sharpness_data, 
            AVERAGING_TARGET, 
            CUSTOM_TARGET_SHARPNESS
        )
        
        self.logger.info(f"目标清晰度: {target_sharpness:.4f}")
        
        # 找到参考图像（用于直方图匹配）
        reference_image_path = None
        current_method = AVERAGING_METHOD
        if current_method == "histogram_matching":
            reference_image_path = find_reference_image(sharpness_data, target_sharpness)
            if reference_image_path:
                self.logger.info(f"参考图像: {Path(reference_image_path).name}")
            else:
                self.logger.warning("未找到合适的参考图像，将使用自适应均衡化方法")
                current_method = "adaptive_equalization"
        
        # 处理每张图像
        total_images = len(sharpness_data)
        for i, (img_path, metrics) in enumerate(sharpness_data.items()):
            try:
                self.logger.info(f"处理图像 {i+1}/{total_images}: {Path(img_path).name}")
                
                # 读取图像
                image = self._read_image(img_path)
                if image is None:
                    continue
                
                # 应用清晰度平均化
                processed_image = self._process_single_image(
                    image, 
                    target_sharpness, 
                    reference_image_path,
                    current_method
                )
                
                # 保存到对应数据集目录
                self._save_to_dataset(processed_image, img_path)
                
            except Exception as e:
                self.logger.error(f"处理图像 {img_path} 失败: {e}")
        
        self.logger.info("清晰度平均化处理完成")
    
    def _read_image(self, img_path: str) -> Optional[np.ndarray]:
        """读取图像"""
        try:
            # 读取图像
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if image is None:
                self.logger.warning(f"无法读取图像: {img_path}")
                return None
            
            # 转换为RGB格式
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 归一化到[0, 1]
            image = image.astype(np.float32) / 255.0
            
            return image
            
        except Exception as e:
            self.logger.error(f"读取图像失败: {e}")
            return None
    
    def _process_single_image(self, image: np.ndarray, target_sharpness: float, 
                             reference_image_path: Optional[str] = None, 
                             method: str = None) -> np.ndarray:
        """处理单张图像"""
        # 准备参考图像（如果使用直方图匹配）
        reference_image = None
        if reference_image_path and method == "histogram_matching":
            reference_image = self._read_image(reference_image_path)
        
        # 应用清晰度平均化
        processed_image = apply_sharpness_averaging(
            image=image,
            target_sharpness=target_sharpness,
            method=method,
            reference_image=reference_image
        )
        
        return processed_image
    
    def _save_to_dataset(self, image: np.ndarray, original_path: str) -> None:
        """保存图像到对应数据集目录"""
        # 确定数据集名称
        dataset_name = self._determine_dataset_name(original_path)
        
        if dataset_name in DATASET_OUTPUT_PATHS:
            output_dir = DATASET_OUTPUT_PATHS[dataset_name]
            
            # 保存图像
            output_path = save_image_to_dataset(
                image=image,
                original_path=original_path,
                dataset_name=dataset_name,
                output_dir=output_dir
            )
            
            self.logger.info(f"图像已保存到: {output_path}")
        else:
            self.logger.warning(f"未找到数据集 {dataset_name} 的输出路径")
    
    def _determine_dataset_name(self, image_path: str) -> str:
        """根据图像路径确定数据集名称"""
        # 从路径中提取数据集名称
        path_parts = Path(image_path).parts
        
        # 添加调试信息
        self.logger.debug(f"解析图像路径: {image_path}")
        self.logger.debug(f"路径部分: {path_parts}")
        
        # 从配置文件导入数据集名称列表
        try:
            from configs.image_sharpness.sharpness_processor_config import DATASET_NAMES
        except ImportError:
            # 如果导入失败，使用硬编码的默认列表
            DATASET_NAMES = ["dataset1_LInCl", "dataset2_LPSCl", "dataset3_LNOCl"]
        
        # 查找包含"dataset"的路径部分，优先查找完整的数据集名称
        for part in path_parts:
            if "dataset" in part.lower():
                # 检查是否是完整的数据集名称
                if any(dataset_name in part for dataset_name in DATASET_NAMES):
                    self.logger.debug(f"从路径部分识别到完整数据集: {part}")
                    return part
                elif part == "datasets":
                    # 如果只是"datasets"目录，继续查找
                    continue
        
        # 如果没有找到完整的数据集名称，尝试从文件名推断
        filename = Path(image_path).name
        self.logger.debug(f"从文件名推断: {filename}")
        
        # 根据文件名特征识别数据集
        if "4.2V" in filename or "4.4V" in filename:
            self.logger.debug(f"识别为 dataset1_LInCl (基于文件名: {filename})")
            return "dataset1_LInCl"
        elif "LPSCl" in filename:
            self.logger.debug(f"识别为 dataset2_LPSCl (基于文件名: {filename})")
            return "dataset2_LPSCl"
        elif "LNOCl" in filename:
            self.logger.debug(f"识别为 dataset3_LNOCl (基于文件名: {filename})")
            return "dataset3_LNOCl"
        else:
            # 如果还是无法识别，尝试从完整路径推断
            path_str = str(image_path)
            self.logger.debug(f"从完整路径推断: {path_str}")
            
            if "dataset1" in path_str or "LInCl" in path_str:
                self.logger.debug(f"识别为 dataset1_LInCl (基于完整路径)")
                return "dataset1_LInCl"
            elif "dataset2" in path_str or "LPSCl" in path_str:
                self.logger.debug(f"识别为 dataset2_LPSCl (基于完整路径)")
                return "dataset2_LPSCl"
            elif "dataset3" in path_str or "LNOCl" in path_str:
                self.logger.debug(f"识别为 dataset3_LNOCl (基于完整路径)")
                return "dataset3_LNOCl"
            else:
                # 最后默认返回dataset1
                self.logger.warning(f"无法识别数据集名称，使用默认值 dataset1_LInCl: {image_path}")
                return "dataset1_LInCl"
    
    def _assess_processed_images(self) -> Dict[str, Dict[str, float]]:
        """评估处理后的图像清晰度"""
        # 收集所有处理后的图像路径
        processed_images = []
        for dataset_name, output_dir in DATASET_OUTPUT_PATHS.items():
            if output_dir.exists():
                for img_file in output_dir.glob("*.png"):
                    processed_images.append(str(img_file))
        
        if not processed_images:
            self.logger.warning("没有找到处理后的图像")
            return {}
        
        # 创建临时文件夹用于评估
        temp_dir = Path("temp_processed_images")
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # 创建符号链接到临时文件夹
            for img_path in processed_images:
                if Path(img_path).exists():
                    temp_path = temp_dir / Path(img_path).name
                    if not temp_path.exists():
                        temp_path.symlink_to(Path(img_path).absolute())
            
            # 评估清晰度
            sharpness_data = self.assessor.assess_batch(temp_dir)
            
            self.logger.info(f"成功评估 {len(sharpness_data)} 张处理后的图像")
            return sharpness_data
            
        finally:
            # 清理临时文件夹
            import shutil
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
    
    def _generate_processed_analysis_charts(self, original_data: Dict[str, Dict[str, float]], 
                                          processed_data: Dict[str, Dict[str, float]]) -> None:
        """生成处理后清晰度分析图表"""
        chart_dir = CHART_OUTPUT_FOLDER / "after_processing"
        chart_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成处理前后对比图
        self.chart_generator.generate_before_after_comparison(
            original_data, 
            processed_data,
            save_path=chart_dir / "before_after_comparison.png"
        )
        
        # 生成处理后清晰度一致性分析
        self.chart_generator.generate_sharpness_consistency_analysis(
            processed_data,
            target_sharpness=TARGET_SHARPNESS,
            save_path=chart_dir / "sharpness_consistency.png"
        )
        
        # 生成处理后的统计图表
        self.chart_generator.generate_sharpness_statistics(
            processed_data,
            save_path=chart_dir / "processed_statistics.png"
        )
        
        self.logger.info(f"处理后清晰度分析图表已保存到: {chart_dir}")
    
    def _generate_training_jsons(self) -> None:
        """生成训练用JSON文件"""
        try:
            # 为每个数据集生成JSON文件
            for dataset_name, output_dir in DATASET_OUTPUT_PATHS.items():
                if output_dir.exists():
                    self._generate_dataset_json(dataset_name, output_dir)
            
            # 生成合并的JSON文件
            self._generate_master_jsons()
            
        except Exception as e:
            self.logger.error(f"生成训练JSON文件失败: {e}")
    
    def _generate_dataset_json(self, dataset_name: str, output_dir: Path) -> None:
        """为单个数据集生成JSON文件"""
        # 收集图像文件
        image_files = list(output_dir.glob("*.png"))
        
        if not image_files:
            self.logger.warning(f"数据集 {dataset_name} 中没有图像文件")
            return
        
        # 生成有监督训练JSON
        if GENERATE_TRAINING_JSON:
            # 查找对应的掩码文件
            mask_dir = Path(f"datasets/{dataset_name}/masks_3class")
            if mask_dir.exists():
                self._generate_supervised_json(dataset_name, output_dir, mask_dir, image_files)
        
        # 生成SSL预训练JSON
        if GENERATE_SSL_JSON:
            self._generate_ssl_json(dataset_name, output_dir, image_files)
    
    def _generate_supervised_json(self, dataset_name: str, image_dir: Path, 
                                 mask_dir: Path, image_files: List[Path]) -> None:
        """生成有监督训练JSON"""
        samples = []
        
        for img_file in image_files:
            # 查找对应的掩码文件
            mask_file = mask_dir / img_file.name
            if mask_file.exists():
                sample = {
                    'dataset': dataset_name,
                    'frames': [str(img_file.relative_to(image_dir))],
                    'mask_file': str(mask_file.relative_to(mask_dir))
                }
                samples.append(sample)
        
        if samples:
            json_data = {
                'samples': samples,
                'dataset_name': dataset_name,
                'root_raw_image_dir': str(image_dir),
                'root_labeled_mask_dir': str(mask_dir),
                'description': f'{dataset_name} 清晰度平均化后的有监督分割训练数据',
                'num_samples': len(samples)
            }
            
            output_path = Path("json") / f"{dataset_name}_sharpness_averaged.json"
            output_path.parent.mkdir(exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"已生成有监督JSON: {output_path}，共{len(samples)}张图片")
    
    def _generate_ssl_json(self, dataset_name: str, image_dir: Path, 
                           image_files: List[Path]) -> None:
        """生成SSL预训练JSON"""
        samples = []
        
        for img_file in image_files:
            sample = {
                'dataset': dataset_name,
                'image_file': str(img_file.relative_to(image_dir))
            }
            samples.append(sample)
        
        if samples:
            json_data = {
                'samples': samples,
                'dataset_name': dataset_name,
                'root_raw_image_dir': str(image_dir),
                'description': f'{dataset_name} 清晰度平均化后的SSL预训练数据',
                'num_samples': len(samples)
            }
            
            output_path = Path("json") / f"{dataset_name}_sharpness_averaged_ssl.json"
            output_path.parent.mkdir(exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"已生成SSL JSON: {output_path}，共{len(samples)}张图片")
    
    def _generate_master_jsons(self) -> None:
        """生成合并的JSON文件"""
        try:
            self.logger.info("开始生成合并的master JSON文件...")
            
            # 收集所有数据集的样本
            all_supervised_samples = []
            all_ssl_samples = []
            datasets_info = {}
            
            # 统计信息
            total_supervised = 0
            total_ssl = 0
            
            for dataset_name, output_dir in DATASET_OUTPUT_PATHS.items():
                self.logger.info(f"处理数据集: {dataset_name}")
                
                if not output_dir.exists():
                    self.logger.warning(f"数据集 {dataset_name} 的输出目录不存在: {output_dir}")
                    continue
                
                # 收集有监督样本
                mask_dir = Path(f"datasets/{dataset_name}/masks_3class")
                supervised_count = 0
                ssl_count = 0
                
                if mask_dir.exists():
                    image_files = list(output_dir.glob("*.png"))
                    for img_file in image_files:
                        mask_file = mask_dir / img_file.name
                        if mask_file.exists():
                            all_supervised_samples.append({
                                'dataset': dataset_name,
                                'frames': [str(img_file.relative_to(output_dir))],
                                'mask_file': str(mask_file.relative_to(mask_dir))
                            })
                            supervised_count += 1
                        else:
                            self.logger.debug(f"图像 {img_file.name} 没有对应的mask文件")
                
                # 收集SSL样本（所有图像都可以用于SSL）
                image_files = list(output_dir.glob("*.png"))
                for img_file in image_files:
                    all_ssl_samples.append({
                        'dataset': dataset_name,
                        'image_file': str(img_file.relative_to(output_dir))
                    })
                    ssl_count += 1
                
                # 记录数据集信息
                datasets_info[dataset_name] = {
                    'raw_image_root': str(output_dir),
                    'mask_root': str(mask_dir) if mask_dir.exists() else None,
                    'supervised_count': supervised_count,
                    'ssl_count': ssl_count,
                    'total_images': len(image_files)
                }
                
                total_supervised += supervised_count
                total_ssl += ssl_count
                
                self.logger.info(f"数据集 {dataset_name}: 有监督样本 {supervised_count}, SSL样本 {ssl_count}, 总图像 {len(image_files)}")
            
            # 生成合并的有监督JSON
            if all_supervised_samples:
                master_supervised_json = Path("json") / "master_sharpness_averaged_dataset.json"
                master_supervised_json.parent.mkdir(exist_ok=True)
                
                supervised_data = {
                    'samples': all_supervised_samples,
                    'datasets_info': datasets_info,
                    'description': '所有数据集清晰度平均化后的有监督分割训练样本',
                    'num_samples': len(all_supervised_samples),
                    'generation_time': datetime.now().isoformat(),
                    'total_supervised': total_supervised,
                    'total_ssl': total_ssl
                }
                
                with open(master_supervised_json, 'w', encoding='utf-8') as f:
                    json.dump(supervised_data, f, indent=2, ensure_ascii=False)
                
                self.logger.info(f"✅ 已生成合并有监督JSON: {master_supervised_json}")
                self.logger.info(f"   包含 {len(all_supervised_samples)} 个有监督样本")
                self.logger.info(f"   来自 {len(datasets_info)} 个数据集")
            else:
                self.logger.warning("⚠️ 没有找到有监督样本，跳过生成有监督JSON")
            
            # 生成合并的SSL JSON
            if all_ssl_samples:
                master_ssl_json = Path("json") / "master_sharpness_averaged_ssl_dataset.json"
                master_ssl_json.parent.mkdir(exist_ok=True)
                
                ssl_data = {
                    'samples': all_ssl_samples,
                    'datasets_info': {k: v['raw_image_root'] for k, v in datasets_info.items()},
                    'description': '所有数据集清晰度平均化后的SSL预训练样本',
                    'num_samples': len(all_ssl_samples),
                    'generation_time': datetime.now().isoformat(),
                    'total_ssl': total_ssl
                }
                
                with open(master_ssl_json, 'w', encoding='utf-8') as f:
                    json.dump(ssl_data, f, indent=2, ensure_ascii=False)
                
                self.logger.info(f"✅ 已生成合并SSL JSON: {master_ssl_json}")
                self.logger.info(f"   包含 {len(all_ssl_samples)} 个SSL样本")
                self.logger.info(f"   来自 {len(datasets_info)} 个数据集")
            else:
                self.logger.warning("⚠️ 没有找到SSL样本，跳过生成SSL JSON")
            
            # 生成处理摘要
            summary = {
                'total_datasets': len(datasets_info),
                'total_supervised_samples': total_supervised,
                'total_ssl_samples': total_ssl,
                'datasets_breakdown': datasets_info,
                'generation_time': datetime.now().isoformat()
            }
            
            summary_json = Path("json") / "sharpness_processing_summary.json"
            with open(summary_json, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"📊 已生成处理摘要: {summary_json}")
            self.logger.info("🎉 Master JSON文件生成完成！")
                
        except Exception as e:
            self.logger.error(f"❌ 生成合并JSON文件失败: {e}")
            import traceback
            self.logger.error(f"详细错误信息: {traceback.format_exc()}")
            raise
    
    def _validate_generated_jsons(self) -> bool:
        """验证生成的JSON文件是否正确"""
        try:
            self.logger.info("开始验证生成的JSON文件...")
            
            validation_results = {}
            
            # 验证有监督JSON
            supervised_json = Path("json") / "master_sharpness_averaged_dataset.json"
            if supervised_json.exists():
                with open(supervised_json, 'r', encoding='utf-8') as f:
                    supervised_data = json.load(f)
                
                # 检查基本结构
                required_keys = ['samples', 'datasets_info', 'num_samples']
                structure_valid = all(key in supervised_data for key in required_keys)
                
                # 检查样本数量
                samples_count = len(supervised_data.get('samples', []))
                declared_count = supervised_data.get('num_samples', 0)
                count_valid = samples_count == declared_count
                
                # 检查数据集信息
                datasets_valid = len(supervised_data.get('datasets_info', {})) > 0
                
                validation_results['supervised'] = {
                    'exists': True,
                    'structure_valid': structure_valid,
                    'count_valid': count_valid,
                    'datasets_valid': datasets_valid,
                    'sample_count': samples_count,
                    'declared_count': declared_count
                }
                
                self.logger.info(f"有监督JSON验证: 结构{'✅' if structure_valid else '❌'}, 数量{'✅' if count_valid else '❌'}, 数据集{'✅' if datasets_valid else '❌'}")
            else:
                validation_results['supervised'] = {'exists': False}
                self.logger.warning("有监督JSON文件不存在")
            
            # 验证SSL JSON
            ssl_json = Path("json") / "master_sharpness_averaged_ssl_dataset.json"
            if ssl_json.exists():
                with open(ssl_json, 'r', encoding='utf-8') as f:
                    ssl_data = json.load(f)
                
                # 检查基本结构
                required_keys = ['samples', 'datasets_info', 'num_samples']
                structure_valid = all(key in ssl_data for key in required_keys)
                
                # 检查样本数量
                samples_count = len(ssl_data.get('samples', []))
                declared_count = ssl_data.get('num_samples', 0)
                count_valid = samples_count == declared_count
                
                # 检查数据集信息
                datasets_valid = len(ssl_data.get('datasets_info', {})) > 0
                
                validation_results['ssl'] = {
                    'exists': True,
                    'structure_valid': structure_valid,
                    'count_valid': count_valid,
                    'datasets_valid': datasets_valid,
                    'sample_count': samples_count,
                    'declared_count': declared_count
                }
                
                self.logger.info(f"SSL JSON验证: 结构{'✅' if structure_valid else '❌'}, 数量{'✅' if count_valid else '❌'}, 数据集{'✅' if datasets_valid else '❌'}")
            else:
                validation_results['ssl'] = {'exists': False}
                self.logger.warning("SSL JSON文件不存在")
            
            # 验证摘要JSON
            summary_json = Path("json") / "sharpness_processing_summary.json"
            if summary_json.exists():
                with open(summary_json, 'r', encoding='utf-8') as f:
                    summary_data = json.load(f)
                
                # 检查基本结构
                required_keys = ['total_datasets', 'total_supervised_samples', 'total_ssl_samples']
                structure_valid = all(key in summary_data for key in required_keys)
                
                validation_results['summary'] = {
                    'exists': True,
                    'structure_valid': structure_valid
                }
                
                self.logger.info(f"摘要JSON验证: 结构{'✅' if structure_valid else '❌'}")
            else:
                validation_results['summary'] = {'exists': False}
                self.logger.warning("摘要JSON文件不存在")
            
            # 总体验证结果
            all_valid = all(
                result.get('exists', False) and 
                result.get('structure_valid', False) and 
                result.get('count_valid', True) and 
                result.get('datasets_valid', True)
                for result in validation_results.values()
                if result.get('exists', False)
            )
            
            if all_valid:
                self.logger.info("🎉 所有JSON文件验证通过！")
            else:
                self.logger.warning("⚠️ 部分JSON文件验证失败，请检查生成的文件")
            
            return all_valid
            
        except Exception as e:
            self.logger.error(f"❌ JSON文件验证失败: {e}")
            return False
    
    def get_processing_summary(self) -> Dict:
        """获取处理摘要"""
        summary = {
            "input_folder": str(INPUT_FOLDER),
            "output_folders": {name: str(path) for name, path in DATASET_OUTPUT_PATHS.items()},
            "total_processed": len(self.processed_images) if hasattr(self, 'processed_images') else 0,
            "chart_output": str(CHART_OUTPUT_FOLDER),
            "json_output": str(JSON_OUTPUT_PATH),
            "sharpness_averaging_enabled": ENABLE_SHARPNESS_AVERAGING,
            "averaging_method": AVERAGING_METHOD,
            "target_sharpness": TARGET_SHARPNESS
        }
        
        # 统计每个数据集的图像数量
        for dataset_name, output_dir in DATASET_OUTPUT_PATHS.items():
            if output_dir.exists():
                image_count = len(list(output_dir.glob("*.png")))
                summary[f"{dataset_name}_image_count"] = image_count
        
        return summary


def main():
    """主函数"""
    try:
        # 创建处理器
        processor = ImageSharpnessProcessor()
        
        # 处理图像
        processor.process_images()
        
        # 打印摘要
        summary = processor.get_processing_summary()
        print("\n" + "="*50)
        print("图像清晰度平均化处理完成！")
        print("="*50)
        print("处理摘要:")
        for key, value in summary.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
        print("="*50)
        
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
