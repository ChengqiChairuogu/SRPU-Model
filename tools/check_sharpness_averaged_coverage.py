#!/usr/bin/env python3
"""
检查清晰度平均化图像覆盖率的程序
对比 sharpness_averaged 和 raw_images 目录中的图像，找出缺失的图像
"""

import os
from pathlib import Path
import json
from collections import defaultdict
import argparse

def get_image_files(directory):
    """获取目录中的所有图像文件"""
    image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.bmp'}
    image_files = set()
    
    if not os.path.exists(directory):
        print(f"警告: 目录不存在: {directory}")
        return image_files
    
    for file_path in Path(directory).rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            # 只保留文件名，不包含路径
            image_files.add(file_path.name)
    
    return image_files

def check_dataset_coverage(dataset_name, raw_images_dir, sharpness_averaged_dir):
    """检查单个数据集的图像覆盖率"""
    print(f"\n{'='*60}")
    print(f"检查数据集: {dataset_name}")
    print(f"{'='*60}")
    
    # 获取原始图像和清晰度平均化图像
    raw_images = get_image_files(raw_images_dir)
    sharpness_averaged = get_image_files(sharpness_averaged_dir)
    
    print(f"原始图像目录: {raw_images_dir}")
    print(f"清晰度平均化目录: {sharpness_averaged_dir}")
    print(f"原始图像数量: {len(raw_images)}")
    print(f"清晰度平均化图像数量: {len(sharpness_averaged)}")
    
    # 找出缺失的图像
    missing_images = raw_images - sharpness_averaged
    extra_images = sharpness_averaged - raw_images
    
    print(f"\n缺失的清晰度平均化图像数量: {len(missing_images)}")
    print(f"多余的清晰度平均化图像数量: {len(extra_images)}")
    
    # 计算覆盖率
    if raw_images:
        coverage = (len(sharpness_averaged) / len(raw_images)) * 100
        print(f"覆盖率: {coverage:.2f}%")
    else:
        coverage = 0
        print("覆盖率: 0% (无原始图像)")
    
    # 显示缺失的图像（前20个）
    if missing_images:
        print(f"\n缺失的图像 (显示前20个):")
        for i, img in enumerate(sorted(missing_images)[:20]):
            print(f"  {i+1:3d}. {img}")
        if len(missing_images) > 20:
            print(f"  ... 还有 {len(missing_images) - 20} 个缺失图像")
    
    # 显示多余的图像（前20个）
    if extra_images:
        print(f"\n多余的图像 (显示前20个):")
        for i, img in enumerate(sorted(extra_images)[:20]):
            print(f"  {i+1:3d}. {img}")
        if len(extra_images) > 20:
            print(f"  ... 还有 {len(extra_images) - 20} 个多余图像")
    
    return {
        'dataset_name': dataset_name,
        'raw_count': len(raw_images),
        'sharpness_count': len(sharpness_averaged),
        'missing_count': len(missing_images),
        'extra_count': len(extra_images),
        'coverage': coverage,
        'missing_images': sorted(missing_images),
        'extra_images': sorted(extra_images)
    }

def check_all_datasets():
    """检查所有数据集"""
    datasets = {
        "dataset1_LInCl": {
            "raw_images": "datasets/dataset1_LInCl/raw_images",
            "sharpness_averaged": "datasets/dataset1_LInCl/sharpness_averaged"
        },
        "dataset2_LPSCl": {
            "raw_images": "datasets/dataset2_LPSCl/raw_images",
            "sharpness_averaged": "datasets/dataset2_LPSCl/sharpness_averaged"
        },
        "dataset3_LNOCl": {
            "raw_images": "datasets/dataset3_LNOCl/raw_images",
            "sharpness_averaged": "datasets/dataset3_LNOCl/sharpness_averaged"
        }
    }
    
    results = []
    total_raw = 0
    total_sharpness = 0
    total_missing = 0
    
    print("清晰度平均化图像覆盖率检查")
    print("=" * 80)
    
    for dataset_name, paths in datasets.items():
        result = check_dataset_coverage(
            dataset_name, 
            paths["raw_images"], 
            paths["sharpness_averaged"]
        )
        results.append(result)
        
        total_raw += result['raw_count']
        total_sharpness += result['sharpness_count']
        total_missing += result['missing_count']
    
    # 总体统计
    print(f"\n{'='*80}")
    print("总体统计")
    print(f"{'='*80}")
    print(f"总原始图像数量: {total_raw}")
    print(f"总清晰度平均化图像数量: {total_sharpness}")
    print(f"总缺失图像数量: {total_missing}")
    
    if total_raw > 0:
        overall_coverage = (total_sharpness / total_raw) * 100
        print(f"总体覆盖率: {overall_coverage:.2f}%")
    else:
        print("总体覆盖率: 0% (无原始图像)")
    
    # 生成详细报告
    generate_detailed_report(results)
    
    return results

def generate_detailed_report(results):
    """生成详细的检查报告"""
    report_file = "sharpness_coverage_report.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("清晰度平均化图像覆盖率检查报告\n")
        f.write("=" * 80 + "\n\n")
        
        for result in results:
            f.write(f"数据集: {result['dataset_name']}\n")
            f.write(f"原始图像数量: {result['raw_count']}\n")
            f.write(f"清晰度平均化图像数量: {result['sharpness_count']}\n")
            f.write(f"缺失图像数量: {result['missing_count']}\n")
            f.write(f"覆盖率: {result['coverage']:.2f}%\n")
            f.write("-" * 40 + "\n")
            
            if result['missing_images']:
                f.write("缺失的图像:\n")
                for img in result['missing_images']:
                    f.write(f"  {img}\n")
            
            if result['extra_images']:
                f.write("多余的图像:\n")
                for img in result['extra_images']:
                    f.write(f"  {img}\n")
            
            f.write("\n")
        
        # 总体统计
        total_raw = sum(r['raw_count'] for r in results)
        total_sharpness = sum(r['sharpness_count'] for r in results)
        total_missing = sum(r['missing_count'] for r in results)
        
        f.write("总体统计\n")
        f.write("-" * 40 + "\n")
        f.write(f"总原始图像数量: {total_raw}\n")
        f.write(f"总清晰度平均化图像数量: {total_sharpness}\n")
        f.write(f"总缺失图像数量: {total_missing}\n")
        
        if total_raw > 0:
            overall_coverage = (total_sharpness / total_raw) * 100
            f.write(f"总体覆盖率: {overall_coverage:.2f}%\n")
    
    print(f"\n详细报告已保存到: {report_file}")

def check_specific_dataset(dataset_name):
    """检查指定的数据集"""
    datasets = {
        "dataset1_LInCl": {
            "raw_images": "datasets/dataset1_LInCl/raw_images",
            "sharpness_averaged": "datasets/dataset1_LInCl/sharpness_averaged"
        },
        "dataset2_LPSCl": {
            "raw_images": "datasets/dataset2_LPSCl/raw_images",
            "sharpness_averaged": "datasets/dataset2_LPSCl/sharpness_averaged"
        },
        "dataset3_LNOCl": {
            "raw_images": "datasets/dataset3_LNOCl/raw_images",
            "sharpness_averaged": "datasets/dataset3_LNOCl/sharpness_averaged"
        }
    }
    
    if dataset_name not in datasets:
        print(f"错误: 未知的数据集名称 '{dataset_name}'")
        print(f"可用的数据集: {list(datasets.keys())}")
        return
    
    paths = datasets[dataset_name]
    result = check_dataset_coverage(dataset_name, paths["raw_images"], paths["sharpness_averaged"])
    
    # 生成该数据集的详细报告
    report_file = f"sharpness_coverage_{dataset_name}_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"数据集 {dataset_name} 清晰度平均化图像覆盖率检查报告\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"原始图像数量: {result['raw_count']}\n")
        f.write(f"清晰度平均化图像数量: {result['sharpness_count']}\n")
        f.write(f"缺失图像数量: {result['missing_count']}\n")
        f.write(f"覆盖率: {result['coverage']:.2f}%\n\n")
        
        if result['missing_images']:
            f.write("缺失的图像:\n")
            for img in result['missing_images']:
                f.write(f"  {img}\n")
        
        if result['extra_images']:
            f.write("\n多余的图像:\n")
            for img in result['extra_images']:
                f.write(f"  {img}\n")
    
    print(f"\n详细报告已保存到: {report_file}")

def main():
    parser = argparse.ArgumentParser(description='检查清晰度平均化图像覆盖率')
    parser.add_argument('--dataset', '-d', type=str, 
                       choices=['dataset1_LInCl', 'dataset2_LPSCl', 'dataset3_LNOCl'],
                       help='指定要检查的数据集，如果不指定则检查所有数据集')
    
    args = parser.parse_args()
    
    if args.dataset:
        check_specific_dataset(args.dataset)
    else:
        check_all_datasets()

if __name__ == "__main__":
    main()
