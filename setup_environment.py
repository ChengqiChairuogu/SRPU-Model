#!/usr/bin/env python3
"""
环境设置和验证脚本
用于检查和安装所需的Python包
"""

import subprocess
import sys
import importlib
from pathlib import Path

def check_package(package_name, import_name=None):
    """检查包是否已安装"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False

def get_package_version(package_name, import_name=None):
    """获取包的版本"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        if hasattr(module, '__version__'):
            return module.__version__
        else:
            return "未知版本"
    except ImportError:
        return "未安装"

def install_package(package_name, version=None):
    """安装包"""
    if version:
        package_spec = f"{package_name}>={version}"
    else:
        package_spec = package_name
    
    try:
        print(f"正在安装 {package_spec}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_spec])
        print(f"✅ {package_name} 安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {package_name} 安装失败: {e}")
        return False

def main():
    """主函数"""
    print("SRPU-Model 环境检查和设置工具")
    print("=" * 50)
    
    # 必需的包列表
    required_packages = {
        # 核心科学计算
        'numpy': {'min_version': '1.21.0', 'import_name': 'numpy'},
        'pandas': {'min_version': '1.3.0', 'import_name': 'pandas'},
        
        # PyTorch生态系统
        'torch': {'min_version': '2.0.0', 'import_name': 'torch'},
        'torchvision': {'min_version': '0.15.0', 'import_name': 'torchvision'},
        'torchaudio': {'min_version': '2.0.0', 'import_name': 'torchaudio'},
        'torchmetrics': {'min_version': '0.11.0', 'import_name': 'torchmetrics'},
        
        # 图像处理
        'opencv-python': {'min_version': '4.5.0', 'import_name': 'cv2'},
        'scikit-image': {'min_version': '0.19.0', 'import_name': 'skimage'},
        'Pillow': {'min_version': '8.0.0', 'import_name': 'PIL'},
        
        # 机器学习和数据科学
        'scikit-learn': {'min_version': '1.0.0', 'import_name': 'sklearn'},
        
        # 可视化
        'matplotlib': {'min_version': '3.5.0', 'import_name': 'matplotlib'},
        'seaborn': {'min_version': '0.11.0', 'import_name': 'seaborn'},
        
        # 工具
        'tqdm': {'min_version': '4.60.0', 'import_name': 'tqdm'},
        'albumentations': {'min_version': '1.3.0', 'import_name': 'albumentations'},
        'wandb': {'min_version': '0.15.0', 'import_name': 'wandb'},
    }
    
    print("检查已安装的包...")
    print()
    
    missing_packages = []
    outdated_packages = []
    
    for package_name, package_info in required_packages.items():
        import_name = package_info['import_name']
        min_version = package_info['min_version']
        
        if check_package(package_name, import_name):
            version = get_package_version(package_name, import_name)
            print(f"✅ {package_name}: {version}")
            
            # 检查版本是否满足要求
            if version != "未知版本" and version != "未安装":
                try:
                    from packaging import version as pkg_version
                    if pkg_version.parse(version) < pkg_version.parse(min_version):
                        outdated_packages.append((package_name, min_version))
                        print(f"   ⚠️  版本过低，需要 >= {min_version}")
                except ImportError:
                    print(f"   ℹ️  无法检查版本兼容性")
        else:
            print(f"❌ {package_name}: 未安装")
            missing_packages.append((package_name, min_version))
    
    print()
    print("=" * 50)
    
    if missing_packages:
        print(f"发现 {len(missing_packages)} 个缺失的包:")
        for package_name, min_version in missing_packages:
            print(f"  - {package_name} (>= {min_version})")
        
        print()
        response = input("是否自动安装缺失的包？(y/n): ").lower().strip()
        
        if response in ['y', 'yes', '是']:
            print("\n开始安装缺失的包...")
            for package_name, min_version in missing_packages:
                install_package(package_name, min_version)
                print()
        else:
            print("\n请手动安装缺失的包:")
            print("pip install " + " ".join([f"{pkg}>={ver}" for pkg, ver in missing_packages]))
    
    if outdated_packages:
        print(f"\n发现 {len(outdated_packages)} 个版本过低的包:")
        for package_name, min_version in outdated_packages:
            print(f"  - {package_name} (需要 >= {min_version})")
        
        print("\n建议更新这些包:")
        print("pip install --upgrade " + " ".join([pkg for pkg, _ in outdated_packages]))
    
    if not missing_packages and not outdated_packages:
        print("🎉 所有必需的包都已正确安装！")
        print("\n您现在可以运行:")
        print("  python tools/image_sharpness_processor.py")
        print("  python -m tasks.ssl_pretrain_task")
    
    print("\n环境检查完成！")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n操作被用户中断")
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()
