# utils/pretrained_model_util.py
import torch
from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime

def load_pretrained_model(
    model,
    checkpoint_path: Path,
    device: torch.device,
    strict: bool = True
) -> Dict[str, Any]:
    """
    加载预训练模型
    
    Args:
        model: 模型实例
        checkpoint_path: 检查点文件路径
        device: 设备
        strict: 是否严格匹配参数
    
    Returns:
        加载的checkpoint信息
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"预训练模型文件不存在: {checkpoint_path}")
    
    print(f"加载预训练模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 检查checkpoint格式
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        print("成功加载模型权重 (model_state_dict格式)")
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=strict)
        print("成功加载模型权重 (state_dict格式)")
    else:
        # 假设checkpoint直接是state_dict
        model.load_state_dict(checkpoint, strict=strict)
        print("成功加载模型权重 (直接state_dict格式)")
    
    return checkpoint

def get_model_info(checkpoint_path: Path) -> Dict[str, Any]:
    """
    获取预训练模型信息
    
    Args:
        checkpoint_path: 检查点文件路径
    
    Returns:
        模型信息字典
    """
    if not checkpoint_path.exists():
        return {"error": "模型文件不存在"}
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    info = {
        "file_path": str(checkpoint_path),
        "file_size_mb": checkpoint_path.stat().st_size / (1024 * 1024),
        "created_time": datetime.fromtimestamp(checkpoint_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 提取模型信息
    if 'epoch' in checkpoint:
        info['epoch'] = checkpoint['epoch']
    if 'val_dice' in checkpoint:
        info['val_dice'] = checkpoint['val_dice']
    if 'train_loss' in checkpoint:
        info['train_loss'] = checkpoint['train_loss']
    if 'config' in checkpoint:
        info['config'] = checkpoint['config']
    
    return info

def list_available_pretrained_models(models_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    列出可用的预训练模型
    
    Args:
        models_dir: 模型目录
    
    Returns:
        模型信息字典
    """
    models_info = {}
    
    if not models_dir.exists():
        return models_info
    
    # 查找所有.pth文件
    for model_file in models_dir.rglob("*.pth"):
        try:
            info = get_model_info(model_file)
            models_info[model_file.name] = info
        except Exception as e:
            models_info[model_file.name] = {"error": str(e)}
    
    return models_info

def save_model_info(checkpoint_path: Path, info: Dict[str, Any]) -> None:
    """
    保存模型信息到JSON文件
    
    Args:
        checkpoint_path: 检查点文件路径
        info: 模型信息
    """
    info_file = checkpoint_path.with_suffix('.json')
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"模型信息已保存到: {info_file}")

def validate_model_compatibility(
    model,
    checkpoint_path: Path,
    device: torch.device
) -> bool:
    """
    验证模型兼容性
    
    Args:
        model: 模型实例
        checkpoint_path: 检查点文件路径
        device: 设备
    
    Returns:
        是否兼容
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 获取模型参数名称
        model_state_dict = model.state_dict()
        model_keys = set(model_state_dict.keys())
        
        # 获取checkpoint参数名称
        if 'model_state_dict' in checkpoint:
            checkpoint_keys = set(checkpoint['model_state_dict'].keys())
        elif 'state_dict' in checkpoint:
            checkpoint_keys = set(checkpoint['state_dict'].keys())
        else:
            checkpoint_keys = set(checkpoint.keys())
        
        # 检查参数匹配
        missing_keys = model_keys - checkpoint_keys
        extra_keys = checkpoint_keys - model_keys
        
        if missing_keys:
            print(f"警告: 模型缺少以下参数: {missing_keys}")
        if extra_keys:
            print(f"警告: checkpoint包含额外参数: {extra_keys}")
        
        return len(missing_keys) == 0
        
    except Exception as e:
        print(f"验证失败: {e}")
        return False

def create_model_summary(models_dir: Path) -> None:
    """
    创建模型摘要
    
    Args:
        models_dir: 模型目录
    """
    print("=" * 60)
    print("预训练模型摘要")
    print("=" * 60)
    
    models_info = list_available_pretrained_models(models_dir)
    
    if not models_info:
        print("没有找到预训练模型")
        return
    
    for model_name, info in models_info.items():
        print(f"\n模型: {model_name}")
        print(f"文件大小: {info.get('file_size_mb', 'N/A'):.2f} MB")
        print(f"创建时间: {info.get('created_time', 'N/A')}")
        
        if 'epoch' in info:
            print(f"训练轮数: {info['epoch']}")
        if 'val_dice' in info:
            print(f"验证Dice: {info['val_dice']:.4f}")
        if 'train_loss' in info:
            print(f"训练损失: {info['train_loss']:.4f}")
        if 'error' in info:
            print(f"错误: {info['error']}")
    
    print("\n" + "=" * 60)

def main():
    """主函数，用于测试"""
    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models" / "checkpoints"
    
    create_model_summary(models_dir)

if __name__ == "__main__":
    main() 