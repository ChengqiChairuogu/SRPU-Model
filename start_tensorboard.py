#!/usr/bin/env python3
"""
TensorBoard 启动脚本
用于快速启动和配置 TensorBoard 服务
"""

import subprocess
import sys
import os
import time
import webbrowser
from pathlib import Path
import argparse

def check_tensorboard_installed():
    """检查 TensorBoard 是否已安装"""
    try:
        import tensorboard
        return True
    except ImportError:
        return False

def find_log_directories():
    """查找可用的日志目录"""
    log_dirs = []
    
    # 检查常见的日志目录
    common_dirs = [
        "runs/tensorboard",
        "runs",
        "logs",
        "tensorboard_logs"
    ]
    
    for dir_path in common_dirs:
        if os.path.exists(dir_path):
            log_dirs.append(dir_path)
    
    # 递归查找包含 tensorboard 的目录
    for root, dirs, files in os.walk("."):
        for dir_name in dirs:
            if "tensorboard" in dir_name.lower():
                full_path = os.path.join(root, dir_name)
                if full_path not in log_dirs:
                    log_dirs.append(full_path)
    
    return log_dirs

def start_tensorboard(log_dir, port=6006, host="localhost", auto_open=True):
    """启动 TensorBoard 服务"""
    
    if not check_tensorboard_installed():
        print("❌ TensorBoard 未安装，正在安装...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorboard"])
            print("✅ TensorBoard 安装成功")
        except subprocess.CalledProcessError:
            print("❌ TensorBoard 安装失败")
            return False
    
    if not os.path.exists(log_dir):
        print(f"❌ 日志目录不存在: {log_dir}")
        return False
    
    print(f"🚀 启动 TensorBoard...")
    print(f"   日志目录: {log_dir}")
    print(f"   端口: {port}")
    print(f"   主机: {host}")
    print(f"   访问地址: http://{host}:{port}")
    print()
    
    # 构建命令
    cmd = [
        sys.executable, "-m", "tensorboard.main",
        "--logdir", log_dir,
        "--port", str(port),
        "--host", host,
        "--reload_interval", "5"
    ]
    
    try:
        # 启动 TensorBoard
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 等待服务启动
        print("⏳ 等待服务启动...")
        time.sleep(3)
        
        # 检查服务是否成功启动
        if process.poll() is None:
            print("✅ TensorBoard 启动成功！")
            
            if auto_open:
                print("🌐 正在打开浏览器...")
                webbrowser.open(f"http://{host}:{port}")
            
            print("\n📊 TensorBoard 正在运行中...")
            print("按 Ctrl+C 停止服务")
            
            try:
                # 等待用户中断
                process.wait()
            except KeyboardInterrupt:
                print("\n🛑 正在停止 TensorBoard...")
                process.terminate()
                process.wait()
                print("✅ TensorBoard 已停止")
            
            return True
        else:
            stdout, stderr = process.communicate()
            print(f"❌ TensorBoard 启动失败:")
            print(f"   错误: {stderr.decode()}")
            return False
            
    except Exception as e:
        print(f"❌ 启动 TensorBoard 时发生错误: {e}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="启动 TensorBoard 服务")
    parser.add_argument("--logdir", "-l", help="日志目录路径")
    parser.add_argument("--port", "-p", type=int, default=6006, help="端口号 (默认: 6006)")
    parser.add_argument("--host", default="localhost", help="主机地址 (默认: localhost)")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    parser.add_argument("--list", action="store_true", help="列出可用的日志目录")
    
    args = parser.parse_args()
    
    print("TensorBoard 启动工具")
    print("=" * 50)
    
    # 列出可用的日志目录
    if args.list:
        log_dirs = find_log_directories()
        if log_dirs:
            print("📁 可用的日志目录:")
            for i, dir_path in enumerate(log_dirs, 1):
                print(f"  {i}. {dir_path}")
            print()
        else:
            print("❌ 未找到可用的日志目录")
            return
    
    # 确定日志目录
    log_dir = args.logdir
    if not log_dir:
        # 尝试自动找到日志目录
        log_dirs = find_log_directories()
        if log_dirs:
            log_dir = log_dirs[0]
            print(f"🔍 自动选择日志目录: {log_dir}")
        else:
            print("❌ 未找到日志目录，请使用 --logdir 参数指定")
            return
    
    # 启动 TensorBoard
    auto_open = not args.no_browser
    success = start_tensorboard(log_dir, args.port, args.host, auto_open)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n操作被用户中断")
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()
