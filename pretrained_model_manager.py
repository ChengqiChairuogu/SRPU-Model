#!/usr/bin/env python3
# pretrained_model_manager.py
"""
预训练模型管理工具

用于查看、验证和管理预训练模型
"""

import sys
from pathlib import Path
import argparse

# 添加项目根目录到sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.pretrained_model_util import (
    list_available_pretrained_models,
    get_model_info,
    validate_model_compatibility,
    create_model_summary
)

def main():
    parser = argparse.ArgumentParser(description="预训练模型管理工具")
    parser.add_argument(
        "action",
        choices=["list", "info", "validate", "summary"],
        help="操作类型: list(列出模型), info(显示模型信息), validate(验证模型), summary(创建摘要)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        help="指定模型文件名（用于info和validate操作）"
    )
    parser.add_argument(
        "--models-dir", "-d",
        type=str,
        default="models/checkpoints",
        help="模型目录路径"
    )
    
    args = parser.parse_args()
    
    models_dir = project_root / args.models_dir
    
    if args.action == "list":
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
    
    elif args.action == "info":
        if args.model is None:
            print("请指定模型文件名: --model <文件名>")
            return
        
        model_path = models_dir / args.model
        if not model_path.exists():
            print(f"模型文件不存在: {model_path}")
            return
        
        info = get_model_info(model_path)
        print(f"\n模型信息: {args.model}")
        print("=" * 40)
        for key, value in info.items():
            if key != 'config':
                print(f"{key}: {value}")
        
        if 'config' in info:
            print(f"\n模型配置:")
            for key, value in info['config'].items():
                print(f"  {key}: {value}")
    
    elif args.action == "validate":
        if args.model is None:
            print("请指定模型文件名: --model <文件名>")
            return
        
        model_path = models_dir / args.model
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
                print(f"模型 {args.model} 与当前架构兼容")
            else:
                print(f"模型 {args.model} 与当前架构不兼容")
                
        except Exception as e:
            print(f"验证过程中出错: {e}")
    
    elif args.action == "summary":
        create_model_summary(models_dir)

if __name__ == "__main__":
    main() 