#!/usr/bin/env python3
# pretrained_model_manager.py
"""
预训练模型管理工具

用于查看、验证和管理预训练模型
支持通过配置文件进行配置
"""

import sys
from pathlib import Path
import argparse

# 添加项目根目录到sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.pretrained_model_util import (
    list_available_pretrained_models,
    get_model_info,
    validate_model_compatibility,
    create_model_summary
)

# 导入配置参数
try:
    from configs.pretrained_model_config import (
        ACTION,
        MODEL_NAME,
        MODELS_DIR,
        AUTO_VALIDATE,
        CREATE_SUMMARY,
        OUTPUT_FORMAT
    )
except ImportError:
    # 如果配置文件不存在，使用默认配置
    ACTION = "list"
    MODEL_NAME = None
    MODELS_DIR = "models/checkpoints"
    AUTO_VALIDATE = False
    CREATE_SUMMARY = False
    OUTPUT_FORMAT = "text"

def main():
    parser = argparse.ArgumentParser(description="预训练模型管理工具")
    parser.add_argument(
        "--action", "-a",
        type=str,
        choices=["list", "info", "validate", "summary"],
        help="操作类型（可选，如果配置文件中已指定则忽略）"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        help="指定模型文件名（可选，如果配置文件中已指定则忽略）"
    )
    parser.add_argument(
        "--models-dir", "-d",
        type=str,
        help="模型目录路径（可选，如果配置文件中已指定则忽略）"
    )
    
    args = parser.parse_args()
    
    # 命令行参数优先级高于配置文件
    action = args.action if args.action else ACTION
    model_name = args.model if args.model else MODEL_NAME
    models_dir = Path(args.models_dir) if args.models_dir else (project_root / MODELS_DIR)
    
    print(f"使用配置: 操作={action}, 模型={model_name}, 目录={models_dir}")
    
    if action == "list":
        print("查找可用的预训练模型...")
        models_info = list_available_pretrained_models(models_dir)
        
        if not models_info:
            print("没有找到预训练模型")
            return
        
        print(f"找到 {len(models_info)} 个预训练模型:")
        for model_name, info in models_info.items():
            print(f"  - {model_name}")
            if 'val_dice' in info:
                print(f"    验证Dice: {info['val_dice']:.4f}")
            if 'epoch' in info:
                print(f"    训练轮数: {info['epoch']}")
        
        # 如果配置了自动验证，则验证所有模型
        if AUTO_VALIDATE:
            print("\n开始自动验证所有模型...")
            for model_name in models_info.keys():
                model_path = models_dir / model_name
                try:
                    from models.segmentation_unet import SegmentationUNet
                    from configs import base as cfg_base
                    
                    model = SegmentationUNet(
                        encoder_name="unet",
                        decoder_name="unet",
                        n_channels=cfg_base.INPUT_DEPTH,
                        n_classes=cfg_base.NUM_CLASSES
                    )
                    
                    import torch
                    device = torch.device("cpu")
                    
                    is_compatible = validate_model_compatibility(model, model_path, device)
                    status = "兼容" if is_compatible else "不兼容"
                    print(f"  {model_name}: {status}")
                    
                except Exception as e:
                    print(f"  {model_name}: 验证失败 - {e}")
    
    elif action == "info":
        if model_name is None:
            print("请指定模型文件名（通过配置文件或--model参数）")
            return
        
        model_path = models_dir / model_name
        if not model_path.exists():
            print(f"模型文件不存在: {model_path}")
            return
        
        info = get_model_info(model_path)
        print(f"\n模型信息: {model_name}")
        print("=" * 40)
        for key, value in info.items():
            if key != 'config':
                print(f"{key}: {value}")
        
        if 'config' in info:
            print(f"\n模型配置:")
            for key, value in info['config'].items():
                print(f"  {key}: {value}")
    
    elif action == "validate":
        if model_name is None:
            print("请指定模型文件名（通过配置文件或--model参数）")
            return
        
        model_path = models_dir / model_name
        if not model_path.exists():
            print(f"模型文件不存在: {model_path}")
            return
        
        # 创建模型实例进行验证
        try:
            from models.segmentation_unet import SegmentationUNet
            from configs import base as cfg_base
            
            model = SegmentationUNet(
                encoder_name="unet",
                decoder_name="unet",
                n_channels=cfg_base.INPUT_DEPTH,
                n_classes=cfg_base.NUM_CLASSES
            )
            
            import torch
            device = torch.device("cpu")
            
            is_compatible = validate_model_compatibility(model, model_path, device)
            if is_compatible:
                print(f"模型 {model_name} 与当前架构兼容")
            else:
                print(f"模型 {model_name} 与当前架构不兼容")
                
        except Exception as e:
            print(f"验证过程中出错: {e}")
    
    elif action == "summary":
        create_model_summary(models_dir)
    
    # 如果配置了创建摘要，则自动创建
    if CREATE_SUMMARY and action != "summary":
        print("\n自动创建模型摘要...")
        create_model_summary(models_dir)

if __name__ == "__main__":
    main() 