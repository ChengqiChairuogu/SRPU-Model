#!/usr/bin/env python3
"""
重新生成数据集分割脚本
使用新的分割比例重新生成训练/验证/测试集分割
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

def regenerate_dataset_splits(
    input_json: str = "json/master_sharpness_averaged_dataset.json",
    output_dir: str = "json",
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42
):
    """
    重新生成数据集分割
    
    Args:
        input_json: 输入的master JSON文件路径
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
    """
    
    # 验证分割比例
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"分割比例总和必须为1.0，当前为: {total_ratio}")
    
    print(f"开始重新生成数据集分割...")
    print(f"输入文件: {input_json}")
    print(f"分割比例: 训练{train_ratio:.1%}, 验证{val_ratio:.1%}, 测试{test_ratio:.1%}")
    print(f"随机种子: {seed}")
    
    # 读取输入JSON
    input_path = Path(input_json)
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_json}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = data.get("samples", [])
    datasets_info = data.get("datasets_info", {})
    total_samples = len(samples)
    
    print(f"总样本数: {total_samples}")
    
    # 设置随机种子
    random.seed(seed)
    
    # 生成随机索引
    indices = list(range(total_samples))
    random.shuffle(indices)
    
    # 计算各集合的样本数量
    n_train = int(total_samples * train_ratio)
    n_val = int(total_samples * val_ratio)
    n_test = total_samples - n_train - n_val
    
    print(f"分割结果: 训练{n_train}张, 验证{n_val}张, 测试{n_test}张")
    
    # 分割样本
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 生成训练集JSON
    train_samples = [samples[i] for i in train_indices]
    train_data = {
        "samples": train_samples,
        "datasets_info": datasets_info,
        "description": f"训练集 - 清晰度平均化后的有监督分割训练样本 (分割比例: {train_ratio:.1%})",
        "num_samples": len(train_samples),
        "split_info": {
            "split": "train",
            "ratio": train_ratio,
            "seed": seed
        }
    }
    
    train_json_path = output_path / "master_sharpness_averaged_dataset_train.json"
    with open(train_json_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 训练集JSON已生成: {train_json_path} ({len(train_samples)}张)")
    
    # 生成验证集JSON
    val_samples = [samples[i] for i in val_indices]
    val_data = {
        "samples": val_samples,
        "datasets_info": datasets_info,
        "description": f"验证集 - 清晰度平均化后的有监督分割训练样本 (分割比例: {val_ratio:.1%})",
        "num_samples": len(val_samples),
        "split_info": {
            "split": "val",
            "ratio": val_ratio,
            "seed": seed
        }
    }
    
    val_json_path = output_path / "master_sharpness_averaged_dataset_val.json"
    with open(val_json_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 验证集JSON已生成: {val_json_path} ({len(val_samples)}张)")
    
    # 生成测试集JSON
    test_samples = [samples[i] for i in test_indices]
    test_data = {
        "samples": test_samples,
        "datasets_info": datasets_info,
        "description": f"测试集 - 清晰度平均化后的有监督分割训练样本 (分割比例: {test_ratio:.1%})",
        "num_samples": len(test_samples),
        "split_info": {
            "split": "test",
            "ratio": test_ratio,
            "seed": seed
        }
    }
    
    test_json_path = output_path / "master_sharpness_averaged_dataset_test.json"
    with open(test_json_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 测试集JSON已生成: {test_json_path} ({len(test_samples)}张)")
    
    # 生成分割摘要
    summary_data = {
        "original_file": str(input_json),
        "total_samples": total_samples,
        "split_ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio
        },
        "split_counts": {
            "train": n_train,
            "val": n_val,
            "test": n_test
        },
        "output_files": {
            "train": str(train_json_path),
            "val": str(val_json_path),
            "test": str(test_json_path)
        },
        "random_seed": seed,
        "description": "数据集分割重新生成结果"
    }
    
    summary_path = output_path / "dataset_split_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    print(f"📊 分割摘要已生成: {summary_path}")
    
    # 打印样本分布
    print(f"\n=== 样本分布 ===")
    print(f"训练集: {n_train}张 ({n_train/total_samples:.1%})")
    print(f"验证集: {n_val}张 ({n_val/total_samples:.1%})")
    print(f"测试集: {n_test}张 ({n_test/total_samples:.1%})")
    
    # 按数据集统计
    print(f"\n=== 按数据集统计 ===")
    for split_name, split_samples in [("训练集", train_samples), ("验证集", val_samples), ("测试集", test_samples)]:
        dataset_counts = {}
        for sample in split_samples:
            dataset = sample.get("dataset", "unknown")
            dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
        
        print(f"{split_name}:")
        for dataset, count in dataset_counts.items():
            print(f"  {dataset}: {count}张")
    
    print(f"\n🎉 数据集分割重新生成完成！")
    return {
        "train": train_json_path,
        "val": val_json_path,
        "test": test_json_path,
        "summary": summary_path
    }

def main():
    """主函数"""
    try:
        # 使用新的分割比例重新生成数据集分割
        result = regenerate_dataset_splits(
            input_json="json/master_sharpness_averaged_dataset.json",
            output_dir="json",
            train_ratio=0.7,  # 训练70%
            val_ratio=0.2,    # 验证20%
            test_ratio=0.1,   # 测试10%
            seed=42
        )
        
        print(f"\n生成的文件:")
        for split, path in result.items():
            if split != "summary":
                print(f"  {split}: {path}")
        
        print(f"\n现在您可以:")
        print(f"1. 使用新的验证集JSON进行更全面的评估")
        print(f"2. 重新训练模型使用新的训练集")
        print(f"3. 在测试集上评估最终性能")
        
    except Exception as e:
        print(f"❌ 重新生成数据集分割失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
