"""
分析图表工具
包含各种数据分析和可视化的图表生成功能
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import platform
import matplotlib.font_manager as fm
import warnings

# 忽略字体警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# 设置中文字体支持
def setup_chinese_fonts():
    """设置中文字体支持"""
    system = platform.system()
    
    if system == "Windows":
        # Windows系统字体 - 优先使用微软雅黑
        chinese_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong']
        
        # 尝试直接设置字体
        for font in chinese_fonts:
            try:
                # 检查字体是否可用
                font_path = fm.findfont(font, fallback_to_default=False)
                if font_path != fm.rcParams['font.sans-serif'][0]:
                    plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
                    print(f"成功设置中文字体: {font}")
                    break
            except Exception as e:
                print(f"尝试设置字体 {font} 失败: {e}")
                continue
        
        # 如果上述方法失败，尝试使用matplotlib的字体管理器
        try:
            # 获取系统字体列表
            font_list = fm.findSystemFonts()
            chinese_font_files = []
            
            for font_file in font_list:
                if any(chinese_name in font_file.lower() for chinese_name in ['msyh', 'simhei', 'simsun']):
                    chinese_font_files.append(font_file)
            
            if chinese_font_files:
                # 使用找到的中文字体文件
                font_prop = fm.FontProperties(fname=chinese_font_files[0])
                plt.rcParams['font.sans-serif'] = [font_prop.get_name()] + plt.rcParams['font.sans-serif']
                print(f"使用字体文件设置中文字体: {chinese_font_files[0]}")
        except Exception as e:
            print(f"使用字体文件设置失败: {e}")
            
    elif system == "Darwin":  # macOS
        # macOS系统字体
        chinese_fonts = ['PingFang SC', 'STHeiti', 'Hiragino Sans GB', 'Arial Unicode MS']
    else:  # Linux
        # Linux系统字体
        chinese_fonts = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'DejaVu Sans', 'Liberation Sans']
    
    # 对于非Windows系统，使用原来的逻辑
    if system != "Windows":
        font_found = False
        for font in chinese_fonts:
            try:
                if fm.findfont(font, fallback_to_default=False) != fm.rcParams['font.sans-serif'][0]:
                    plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
                    font_found = True
                    break
            except:
                continue
        
        if not font_found:
            try:
                default_font = fm.FontProperties(family='sans-serif')
                if default_font.get_name() != 'DejaVu Sans':
                    plt.rcParams['font.sans-serif'] = [default_font.get_name()] + plt.rcParams['font.sans-serif']
            except:
                pass
    
    # 设置负号显示
    plt.rcParams['axes.unicode_minus'] = False
    
    # 设置字体大小
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    
    # 强制刷新字体设置
    plt.rcParams.update(plt.rcParams)

def force_chinese_font():
    """强制设置中文字体，确保中文能够显示"""
    try:
        import platform
        system = platform.system()
        
        if system == "Windows":
            # Windows系统强制中文字体设置
            font_candidates = [
                'Microsoft YaHei',
                'SimHei', 
                'SimSun',
                'KaiTi',
                'FangSong'
            ]
            
            # 尝试找到可用的中文字体
            available_font = None
            for font_name in font_candidates:
                try:
                    font_path = fm.findfont(font_name, fallback_to_default=False)
                    if font_path and 'ttf' in font_path.lower():
                        available_font = font_name
                        break
                except:
                    continue
            
            if available_font:
                # 设置字体
                plt.rcParams['font.sans-serif'] = [available_font] + plt.rcParams['font.sans-serif']
                print(f"强制设置中文字体: {available_font}")
                return True
            else:
                # 如果找不到中文字体，尝试使用系统默认字体
                try:
                    # 获取系统字体列表并查找中文字体文件
                    font_files = fm.findSystemFonts()
                    chinese_fonts = []
                    
                    for font_file in font_files:
                        font_name = Path(font_file).stem.lower()
                        if any(name in font_name for name in ['msyh', 'simhei', 'simsun', 'kaiti', 'fangsong']):
                            chinese_fonts.append(font_file)
                    
                    if chinese_fonts:
                        # 使用第一个找到的中文字体文件
                        font_prop = fm.FontProperties(fname=chinese_fonts[0])
                        plt.rcParams['font.sans-serif'] = [font_prop.get_name()] + plt.rcParams['font.sans-serif']
                        print(f"使用字体文件强制设置: {chinese_fonts[0]}")
                        return True
                except Exception as e:
                    print(f"字体文件设置失败: {e}")
        
        # 通用设置
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        print("使用通用字体设置")
        return False
        
    except Exception as e:
        print(f"强制字体设置失败: {e}")
        return False

# 初始化中文字体
setup_chinese_fonts()

class SharpnessAnalysisChartGenerator:
    """清晰度分析图表生成器"""
    
    def __init__(self):
        """初始化图表生成器"""
        self.setup_style()
        self.verify_chinese_fonts()
    
    def setup_style(self):
        """设置图表样式"""
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
    
    def verify_chinese_fonts(self):
        """验证中文字体是否正常工作"""
        try:
            # 测试中文字体
            test_fig, test_ax = plt.subplots(figsize=(1, 1))
            test_ax.set_title('测试中文显示')
            test_ax.text(0.5, 0.5, '中文测试', ha='center', va='center')
            plt.close(test_fig)
            print("中文字体验证成功")
        except Exception as e:
            print(f"中文字体验证失败: {e}")
            # 尝试重新设置字体
            self._fallback_font_setup()
    
    def _fallback_font_setup(self):
        """字体回退设置"""
        try:
            # 对于Windows系统，尝试更强制的中文字体设置
            import platform
            if platform.system() == "Windows":
                # 尝试使用系统字体文件
                import matplotlib.font_manager as fm
                font_list = fm.findSystemFonts()
                
                # 查找中文字体文件
                chinese_fonts = []
                for font_file in font_list:
                    font_name = Path(font_file).stem.lower()
                    if any(name in font_name for name in ['msyh', 'simhei', 'simsun', 'kaiti', 'fangsong']):
                        chinese_fonts.append(font_file)
                
                if chinese_fonts:
                    # 使用第一个找到的中文字体
                    font_prop = fm.FontProperties(fname=chinese_fonts[0])
                    plt.rcParams['font.sans-serif'] = [font_prop.get_name()] + plt.rcParams['font.sans-serif']
                    print(f"使用回退中文字体: {chinese_fonts[0]}")
                    return
                
                # 如果找不到中文字体文件，尝试使用字体名称
                windows_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun']
                for font in windows_fonts:
                    try:
                        if fm.findfont(font, fallback_to_default=False) != fm.rcParams['font.sans-serif'][0]:
                            plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
                            print(f"使用回退字体名称: {font}")
                            break
                    except:
                        continue
            
            # 通用回退设置
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Arial']
            plt.rcParams['axes.unicode_minus'] = False
            print("已使用通用回退字体设置")
            
        except Exception as e:
            print(f"字体回退设置失败: {e}")
    
    def _ensure_chinese_support(self):
        """确保中文支持"""
        try:
            # 检查当前字体是否支持中文
            current_font = plt.rcParams['font.sans-serif'][0]
            
            # 对于Windows系统，检查是否包含中文字体
            import platform
            if platform.system() == "Windows":
                if not any(chinese_font in current_font for chinese_font in ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong']):
                    print(f"当前字体 {current_font} 可能不支持中文，尝试重新设置")
                    # 重新设置中文字体
                    setup_chinese_fonts()
                    
                    # 再次检查
                    new_font = plt.rcParams['font.sans-serif'][0]
                    if new_font != current_font:
                        print(f"字体已更新为: {new_font}")
                    else:
                        print("字体更新失败，使用当前字体")
        except Exception as e:
            print(f"字体检查失败: {e}")
            # 尝试重新设置
            setup_chinese_fonts()
    
    def _force_chinese_font_before_plot(self):
        """在绘图前强制设置中文字体"""
        try:
            # 强制设置中文字体
            if not force_chinese_font():
                # 如果强制设置失败，尝试重新设置
                setup_chinese_fonts()
            
            # 确保设置生效
            plt.rcParams['axes.unicode_minus'] = False
            
        except Exception as e:
            print(f"强制字体设置失败: {e}")
    
    def generate_sharpness_distribution(self, 
                                      sharpness_data: Dict[str, Dict[str, float]], 
                                      save_path: Optional[Path] = None) -> None:
        """生成清晰度分布直方图"""
        # 强制确保中文支持
        self._force_chinese_font_before_plot()
        
        # 提取清晰度分数
        sharpness_scores = [data["sharpness"] for data in sharpness_data.values()]
        
        # 创建子图 - 只创建一个图表对象
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 直方图
        ax1.hist(sharpness_scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('清晰度分数')
        ax1.set_ylabel('图片数量')
        ax1.set_title('图片清晰度分布直方图')
        ax1.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_score = np.mean(sharpness_scores)
        std_score = np.std(sharpness_scores)
        ax1.axvline(mean_score, color='red', linestyle='--', 
                   label=f'平均值: {mean_score:.3f}')
        ax1.axvline(mean_score + std_score, color='orange', linestyle=':', 
                   label=f'平均值+标准差: {mean_score + std_score:.3f}')
        ax1.axvline(mean_score - std_score, color='orange', linestyle=':', 
                   label=f'平均值-标准差: {mean_score - std_score:.3f}')
        ax1.legend()
        
        # 箱线图
        ax2.boxplot(sharpness_scores, vert=False)
        ax2.set_xlabel('清晰度分数')
        ax2.set_title('图片清晰度箱线图')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            try:
                # 确保保存路径的目录存在
                save_path.parent.mkdir(parents=True, exist_ok=True)
                # 保存图表
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"清晰度分布图已保存到: {save_path}")
            except Exception as e:
                print(f"保存图表失败: {e}")
        
        # 显示图表
        plt.show()
        
        # 关闭图表以释放内存
        plt.close(fig)
    
    def generate_sharpness_comparison(self, 
                                    sharpness_data: Dict[str, Dict[str, float]], 
                                    save_path: Optional[Path] = None) -> None:
        """生成不同清晰度指标的对比图"""
        # 强制确保中文支持
        self._force_chinese_font_before_plot()
        
        # 提取不同指标的数据
        metrics = ['lapvar', 'tenengrad', 'fft_energy']
        metric_data = {metric: [] for metric in metrics}
        
        for data in sharpness_data.values():
            for metric in metrics:
                if metric in data:
                    metric_data[metric].append(data[metric])
        
        # 创建子图网格 - 只创建一个图表对象
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 各指标分布对比
        for i, metric in enumerate(metrics):
            row, col = i // 2, i % 2
            axes[row, col].hist(metric_data[metric], bins=25, alpha=0.7, 
                               color=['skyblue', 'lightgreen', 'lightcoral'][i])
            axes[row, col].set_xlabel(f'{metric} 分数')
            axes[row, col].set_ylabel('图片数量')
            axes[row, col].set_title(f'{metric} 分布')
            axes[row, col].grid(True, alpha=0.3)
        
        # 2. 综合清晰度分布
        overall_scores = [data["sharpness"] for data in sharpness_data.values()]
        axes[1, 1].hist(overall_scores, bins=25, alpha=0.7, color='gold')
        axes[1, 1].set_xlabel('综合清晰度分数')
        axes[1, 1].set_ylabel('图片数量')
        axes[1, 1].set_title('综合清晰度分布')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            try:
                # 确保保存路径的目录存在
                save_path.parent.mkdir(parents=True, exist_ok=True)
                # 保存图表
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"清晰度对比图已保存到: {save_path}")
            except Exception as e:
                print(f"保存图表失败: {e}")
        
        # 显示图表
        plt.show()
        
        # 关闭图表以释放内存
        plt.close(fig)
    
    def generate_sharpness_statistics(self, 
                                    sharpness_data: Dict[str, Dict[str, float]], 
                                    save_path: Optional[Path] = None) -> None:
        """生成清晰度统计信息图"""
        # 强制确保中文支持
        self._force_chinese_font_before_plot()
        
        # 计算统计信息
        metrics = ['lapvar', 'tenengrad', 'fft_energy', 'sharpness']
        stats_data = {}
        
        for metric in metrics:
            values = [data.get(metric, 0) for data in sharpness_data.values() if metric in data]
            if values:
                stats_data[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
        
        # 创建统计图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 1. 统计指标条形图
        metrics_names = list(stats_data.keys())
        means = [stats_data[m]['mean'] for m in metrics_names]
        stds = [stats_data[m]['std'] for m in metrics_names]
        
        x_pos = np.arange(len(metrics_names))
        bars = ax1.bar(x_pos, means, yerr=stds, capsize=5, 
                      color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        ax1.set_xlabel('清晰度指标')
        ax1.set_ylabel('分数')
        ax1.set_title('各指标统计信息')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(metrics_names)
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom')
        
        # 2. 箱线图对比
        all_data = []
        labels = []
        for metric in metrics:
            values = [data.get(metric, 0) for data in sharpness_data.values() if metric in data]
            if values:
                all_data.append(values)
                labels.append(metric)
        
        ax2.boxplot(all_data, labels=labels)
        ax2.set_xlabel('清晰度指标')
        ax2.set_ylabel('分数')
        ax2.set_title('各指标箱线图对比')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            try:
                # 确保保存路径的目录存在
                save_path.parent.mkdir(parents=True, exist_ok=True)
                # 保存图表
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"清晰度统计图已保存到: {save_path}")
            except Exception as e:
                print(f"保存图表失败: {e}")
        
        # 显示图表
        plt.show()
        
        # 关闭图表以释放内存
        plt.close(fig)
    
    def generate_before_after_comparison(self, 
                                       original_data: Dict[str, Dict[str, float]], 
                                       optimized_data: Dict[str, Dict[str, float]], 
                                       save_path: Optional[Path] = None) -> None:
        """生成优化前后的对比图"""
        # 强制确保中文支持
        self._force_chinese_font_before_plot()
        
        # 提取清晰度分数
        original_scores = [data["sharpness"] for data in original_data.values()]
        optimized_scores = [data["sharpness"] for data in optimized_data.values()]
        
        # 创建子图 - 只创建一个图表对象
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 优化前后分布对比
        ax1.hist(original_scores, bins=25, alpha=0.7, color='lightcoral', 
                label='优化前', edgecolor='black')
        ax1.hist(optimized_scores, bins=25, alpha=0.7, color='lightgreen', 
                label='优化后', edgecolor='black')
        ax1.set_xlabel('清晰度分数')
        ax1.set_ylabel('图片数量')
        ax1.set_title('优化前后清晰度分布对比')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 优化前后箱线图对比
        ax2.boxplot([original_scores, optimized_scores], labels=['优化前', '优化后'])
        ax2.set_ylabel('清晰度分数')
        ax2.set_title('优化前后清晰度箱线图对比')
        ax2.grid(True, alpha=0.3)
        
        # 3. 改进效果散点图
        # 创建对应的图片对
        improvement_data = []
        for img_path, opt_data in optimized_data.items():
            # 尝试找到对应的原始图片
            img_name = Path(img_path).stem
            for orig_path, orig_data in original_data.items():
                if img_name in orig_path:
                    improvement_data.append({
                        'original': orig_data['sharpness'],
                        'optimized': opt_data['sharpness'],
                        'improvement': opt_data['sharpness'] - orig_data['sharpness']
                    })
                    break
        
        if improvement_data:
            orig_scores = [d['original'] for d in improvement_data]
            opt_scores = [d['optimized'] for d in improvement_data]
            improvements = [d['improvement'] for d in improvement_data]
            
            # 散点图
            scatter = ax3.scatter(orig_scores, opt_scores, c=improvements, 
                                cmap='RdYlGn', alpha=0.7, s=50)
            ax3.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='无改进线')
            ax3.set_xlabel('原始清晰度分数')
            ax3.set_ylabel('优化后清晰度分数')
            ax3.set_title('清晰度改进效果散点图')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 添加颜色条
            cbar = plt.colorbar(scatter, ax=ax3)
            cbar.set_label('改进幅度')
            
            # 4. 改进幅度分布
            ax4.hist(improvements, bins=25, alpha=0.7, color='orange', edgecolor='black')
            ax4.axvline(0, color='red', linestyle='--', label='无改进线')
            ax4.set_xlabel('清晰度改进幅度')
            ax4.set_ylabel('图片数量')
            ax4.set_title('清晰度改进幅度分布')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            try:
                # 确保保存路径的目录存在
                save_path.parent.mkdir(parents=True, exist_ok=True)
                # 保存图表
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"优化前后对比图已保存到: {save_path}")
            except Exception as e:
                print(f"保存图表失败: {e}")
        
        # 显示图表
        plt.show()
        
        # 关闭图表以释放内存
        plt.close(fig)
    
    def generate_sharpness_consistency_analysis(self, 
                                              optimized_data: Dict[str, Dict[str, float]], 
                                              target_sharpness: float = 0.7, 
                                              save_path: Optional[Path] = None) -> None:
        """生成清晰度一致性分析图"""
        # 强制确保中文支持
        self._force_chinese_font_before_plot()
        
        # 提取清晰度分数
        sharpness_scores = [data["sharpness"] for data in optimized_data.values()]
        
        # 创建子图 - 只创建一个图表对象
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 清晰度分布与目标值对比
        ax1.hist(sharpness_scores, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        ax1.axvline(target_sharpness, color='red', linestyle='--', linewidth=2, 
                   label=f'目标清晰度: {target_sharpness}')
        ax1.axvline(target_sharpness + 0.1, color='orange', linestyle=':', alpha=0.7,
                   label=f'目标±0.1范围')
        ax1.axvline(target_sharpness - 0.1, color='orange', linestyle=':', alpha=0.7)
        ax1.set_xlabel('清晰度分数')
        ax1.set_ylabel('图片数量')
        ax1.set_title('优化后清晰度分布与目标值对比')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 与目标值的偏差分布
        deviations = [score - target_sharpness for score in sharpness_scores]
        ax2.hist(deviations, bins=25, alpha=0.7, color='lightblue', edgecolor='black')
        ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='目标值')
        ax2.axvline(0.1, color='orange', linestyle=':', alpha=0.7, label='±0.1范围')
        ax2.axvline(-0.1, color='orange', linestyle=':', alpha=0.7)
        ax2.set_xlabel('与目标值的偏差')
        ax2.set_ylabel('图片数量')
        ax2.set_title('清晰度与目标值偏差分布')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 一致性统计
        within_target = sum(1 for score in sharpness_scores if abs(score - target_sharpness) < 0.1)
        total_images = len(sharpness_scores)
        consistency_rate = within_target / total_images * 100
        
        # 饼图显示一致性
        sizes = [consistency_rate, 100 - consistency_rate]
        labels = [f'达到目标\n({within_target}张)', f'未达到目标\n({total_images - within_target}张)']
        colors = ['lightgreen', 'lightcoral']
        
        ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax3.set_title('清晰度一致性统计')
        
        # 4. 清晰度分数排序图
        sorted_scores = sorted(sharpness_scores)
        ax4.plot(range(len(sorted_scores)), sorted_scores, 'b-', linewidth=2, alpha=0.7)
        ax4.axhline(target_sharpness, color='red', linestyle='--', linewidth=2, 
                   label=f'目标清晰度: {target_sharpness}')
        ax4.axhline(target_sharpness + 0.1, color='orange', linestyle=':', alpha=0.7,
                   label='目标±0.1范围')
        ax4.axhline(target_sharpness - 0.1, color='orange', linestyle=':', alpha=0.7)
        ax4.set_xlabel('图片索引（按清晰度排序）')
        ax4.set_ylabel('清晰度分数')
        ax4.set_title('清晰度分数排序图')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"清晰度一致性分析图已保存到: {save_path}")
        
        plt.show()


class GeneralAnalysisChartGenerator:
    """通用分析图表生成器"""
    
    def __init__(self):
        """初始化图表生成器"""
        self.setup_style()
        self.verify_chinese_fonts()
    
    def setup_style(self):
        """设置图表样式"""
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
    
    def verify_chinese_fonts(self):
        """验证中文字体是否正常工作"""
        try:
            # 测试中文字体
            test_fig, test_ax = plt.subplots(figsize=(1, 1))
            test_ax.set_title('测试中文显示')
            test_ax.text(0.5, 0.5, '中文测试', ha='center', va='center')
            plt.close(test_fig)
            print("中文字体验证成功")
        except Exception as e:
            print(f"中文字体验证失败: {e}")
            # 尝试重新设置字体
            self._fallback_font_setup()
    
    def _fallback_font_setup(self):
        """字体回退设置"""
        try:
            # 对于Windows系统，尝试更强制的中文字体设置
            import platform
            if platform.system() == "Windows":
                # 尝试使用系统字体文件
                import matplotlib.font_manager as fm
                font_list = fm.findSystemFonts()
                
                # 查找中文字体文件
                chinese_fonts = []
                for font_file in font_list:
                    font_name = Path(font_file).stem.lower()
                    if any(name in font_name for name in ['msyh', 'simhei', 'simsun', 'kaiti', 'fangsong']):
                        chinese_fonts.append(font_file)
                
                if chinese_fonts:
                    # 使用第一个找到的中文字体
                    font_prop = fm.FontProperties(fname=chinese_fonts[0])
                    plt.rcParams['font.sans-serif'] = [font_prop.get_name()] + plt.rcParams['font.sans-serif']
                    print(f"使用回退中文字体: {chinese_fonts[0]}")
                    return
                
                # 如果找不到中文字体文件，尝试使用字体名称
                windows_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun']
                for font in windows_fonts:
                    try:
                        if fm.findfont(font, fallback_to_default=False) != fm.rcParams['font.sans-serif'][0]:
                            plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
                            print(f"使用回退字体名称: {font}")
                            break
                    except:
                        continue
            
            # 通用回退设置
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Arial']
            plt.rcParams['axes.unicode_minus'] = False
            print("已使用通用回退字体设置")
            
        except Exception as e:
            print(f"字体回退设置失败: {e}")
    
    def _ensure_chinese_support(self):
        """确保中文支持"""
        try:
            # 检查当前字体是否支持中文
            current_font = plt.rcParams['font.sans-serif'][0]
            
            # 对于Windows系统，检查是否包含中文字体
            import platform
            if platform.system() == "Windows":
                if not any(chinese_font in current_font for chinese_font in ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong']):
                    print(f"当前字体 {current_font} 可能不支持中文，尝试重新设置")
                    # 重新设置中文字体
                    setup_chinese_fonts()
                    
                    # 再次检查
                    new_font = plt.rcParams['font.sans-serif'][0]
                    if new_font != current_font:
                        print(f"字体已更新为: {new_font}")
                    else:
                        print("字体更新失败，使用当前字体")
        except Exception as e:
            print(f"字体检查失败: {e}")
            # 尝试重新设置
            setup_chinese_fonts()
    
    def _force_chinese_font_before_plot(self):
        """在绘图前强制设置中文字体"""
        try:
            # 强制设置中文字体
            if not force_chinese_font():
                # 如果强制设置失败，尝试重新设置
                setup_chinese_fonts()
            
            # 确保设置生效
            plt.rcParams['axes.unicode_minus'] = False
            
        except Exception as e:
            print(f"强制字体设置失败: {e}")
    
    def generate_correlation_matrix(self, data: pd.DataFrame, 
                                  save_path: Optional[Path] = None) -> None:
        """生成相关性矩阵热力图"""
        # 强制确保中文支持
        self._force_chinese_font_before_plot()
        
        plt.figure(figsize=(10, 8))
        correlation_matrix = data.corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('相关性矩阵热力图')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"相关性矩阵图已保存到: {save_path}")
        
        plt.show()
    
    def generate_time_series_plot(self, time_data: List, values: List, 
                                 title: str = "时间序列图",
                                 save_path: Optional[Path] = None) -> None:
        """生成时间序列图"""
        # 强制确保中文支持
        self._force_chinese_font_before_plot()
        
        plt.figure(figsize=(12, 6))
        plt.plot(time_data, values, 'b-', linewidth=2, alpha=0.7)
        plt.xlabel('时间')
        plt.ylabel('数值')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"时间序列图已保存到: {save_path}")
        
        plt.show()
    
    def generate_performance_comparison(self, methods: List[str], 
                                      metrics: List[str], 
                                      performance_data: List[List[float]],
                                      save_path: Optional[Path] = None) -> None:
        """生成性能对比图"""
        # 强制确保中文支持
        self._force_chinese_font_before_plot()
        
        x = np.arange(len(methods))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for i, metric in enumerate(metrics):
            offset = (i - len(metrics)/2 + 0.5) * width
            bars = ax.bar(x + offset, [row[i] for row in performance_data], 
                         width, label=metric, alpha=0.8)
        
        ax.set_xlabel('方法')
        ax.set_ylabel('性能指标')
        ax.set_title('不同方法性能对比')
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"性能对比图已保存到: {save_path}")
        
        plt.show()


# 便捷函数
def create_sharpness_analysis_charts(sharpness_data: Dict[str, Dict[str, float]], 
                                   output_dir: Path) -> None:
    """创建完整的清晰度分析图表"""
    generator = SharpnessAnalysisChartGenerator()
    
    # 生成各种图表
    generator.generate_sharpness_distribution(
        sharpness_data, 
        save_path=output_dir / "sharpness_distribution.png"
    )
    
    generator.generate_sharpness_comparison(
        sharpness_data,
        save_path=output_dir / "sharpness_comparison.png"
    )
    
    generator.generate_sharpness_statistics(
        sharpness_data,
        save_path=output_dir / "sharpness_statistics.png"
    )
    
    print(f"所有清晰度分析图表已保存到: {output_dir}")


def create_before_after_comparison(original_data: Dict[str, Dict[str, float]], 
                                 optimized_data: Dict[str, Dict[str, float]], 
                                 output_dir: Path) -> None:
    """创建优化前后对比图"""
    generator = SharpnessAnalysisChartGenerator()
    
    generator.generate_before_after_comparison(
        original_data, 
        optimized_data,
        save_path=output_dir / "before_after_comparison.png"
    )
    
    print(f"优化前后对比图已保存到: {output_dir}")
