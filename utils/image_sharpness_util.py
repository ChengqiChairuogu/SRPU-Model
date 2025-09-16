"""
utils/image_sharpness_util.py
Simplified utility to assess image sharpness.
"""

from pathlib import Path
from typing import Dict, Sequence, Literal, Union
import cv2
import numpy as np
import torch
from skimage import img_as_float32
from skimage.color import rgb2gray

MetricName = Literal["lapvar", "tenengrad", "fft_energy"]


def _ensure_gray_float(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = rgb2gray(img_as_float32(img))
    else:
        gray = img.astype(np.float32) / (255.0 if img.max() > 1.0 else 1.0)
    return gray


def lapvar(gray: np.ndarray) -> float:
    """计算拉普拉斯方差，增加数值稳定性"""
    try:
        # 检查输入图像是否有效
        if gray.size == 0 or np.all(gray == gray[0, 0]):
            return 0.0
        
        # 计算拉普拉斯算子
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        
        # 计算方差，增加数值稳定性
        variance = float(laplacian.var())
        
        # 检查异常值
        if np.isnan(variance) or np.isinf(variance):
            return 0.0
            
        return variance
    except Exception as e:
        print(f"lapvar计算失败: {e}")
        return 0.0


def tenengrad(gray: np.ndarray) -> float:
    """计算Tenengrad梯度，增加数值稳定性"""
    try:
        # 检查输入图像是否有效
        if gray.size == 0 or np.all(gray == gray[0, 0]):
            return 0.0
        
        # 计算Sobel梯度
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        
        # 计算梯度幅值
        gradient_magnitude = np.sqrt((gx * gx + gy * gy).mean())
        
        # 检查异常值
        if np.isnan(gradient_magnitude) or np.isinf(gradient_magnitude):
            return 0.0
            
        return float(gradient_magnitude)
    except Exception as e:
        print(f"tenengrad计算失败: {e}")
        return 0.0


def fft_energy(gray: np.ndarray, hp_radius: int = 4) -> float:
    """计算FFT高频能量，增加数值稳定性"""
    try:
        # 检查输入图像是否有效
        if gray.size == 0 or np.all(gray == gray[0, 0]):
            return 0.0
        
        h, w = gray.shape
        
        # 计算FFT
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft = np.fft.fftshift(dft)
        mag = cv2.magnitude(dft[..., 0], dft[..., 1])
        
        # 创建高频掩码
        cy, cx = h // 2, w // 2
        mask = (np.ogrid[:h, :w][0] - cy) ** 2 + (np.ogrid[:h, :w][1] - cx) ** 2 > hp_radius ** 2
        
        # 计算高频能量
        if np.any(mask):
            high_freq_energy = float(mag[mask].mean())
            
            # 检查异常值
            if np.isnan(high_freq_energy) or np.isinf(high_freq_energy):
                return 0.0
                
            return high_freq_energy
        else:
            return 0.0
    except Exception as e:
        print(f"fft_energy计算失败: {e}")
        return 0.0


_METRIC_FUNCS = {"lapvar": lapvar, "tenengrad": tenengrad, "fft_energy": fft_energy}


class SharpnessAssessor:
    def __init__(self, methods: Sequence[MetricName] = ("lapvar", "tenengrad")):
        self.methods = methods

    def assess_image(self, path: Union[Path, str]) -> Dict[str, float]:
        gray = _ensure_gray_float(cv2.imread(str(path), cv2.IMREAD_UNCHANGED))
        results = {m: _METRIC_FUNCS[m](gray) for m in self.methods}
        results["sharpness"] = float(np.mean(list(results.values())))
        return results

    def assess_batch(self, folder: Union[Path, str]) -> Dict[str, Dict[str, float]]:
        folder = Path(folder)
        files = [p for p in folder.rglob("*.*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".bmp")]
        return {str(p): self.assess_image(p) for p in files}


def build_degrader(mode: str, **kwargs):
    """
    构建图像降质操作函数
    
    Args:
        mode: 降质模式
        **kwargs: 降质参数
    
    Returns:
        降质操作函数
    """
    if mode == "bicubic_downup":
        scale = kwargs.get("scale", 0.5)
        def degrade_op(images):
            """双三次插值下采样再上采样"""
            import torch.nn.functional as F
            
            # 确保输入是4D tensor [B, C, H, W]
            if images.dim() == 3:
                images = images.unsqueeze(0)
            
            B, C, H, W = images.shape
            new_H, new_W = int(H * scale), int(W * scale)
            
            # 下采样
            downsampled = F.interpolate(images, size=(new_H, new_W), mode='bicubic', align_corners=False)
            # 上采样回原尺寸
            upsampled = F.interpolate(downsampled, size=(H, W), mode='bicubic', align_corners=False)
            
            return upsampled
        
        return degrade_op
            
    elif mode == "gaussian_blur":
        def degrade_op(images):
            """高斯模糊降质 - 简化版本"""
            import torch.nn.functional as F
            
            # 确保输入是4D tensor [B, C, H, W]
            if images.dim() == 3:
                images = images.unsqueeze(0)
            
            B, C, H, W = images.shape
            
            # 获取参数
            kernel_size = kwargs.get("ksize", 5)
            sigma = kwargs.get("sigma", 1.0)
            
            # 确保核大小是奇数
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            # 使用PyTorch内置的高斯模糊
            blurred = F.avg_pool2d(images, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
            
            # 添加额外的模糊效果
            if sigma > 2.0:
                blurred = F.avg_pool2d(blurred, kernel_size=3, stride=1, padding=1)
            
            return blurred
        
        return degrade_op
            
    elif mode == "jpeg_compression":
        quality = kwargs.get("quality", 35)
        def degrade_op(images):
            """JPEG压缩模拟 - 修复版本"""
            import torch.nn.functional as F
            from PIL import Image
            import io
            
            # 确保输入是4D tensor [B, C, H, W]
            if images.dim() == 3:
                images = images.unsqueeze(0)
            
            B, C, H, W = images.shape
            degraded = []
            
            for i in range(B):
                # 转换为PIL图像
                img_tensor = images[i]
                if img_tensor.max() <= 1.0:
                    img_tensor = (img_tensor * 255).clamp(0, 255).byte()
                
                img_pil = Image.fromarray(img_tensor.permute(1, 2, 0).cpu().numpy())
                
                # JPEG压缩
                buffer = io.BytesIO()
                img_pil.save(buffer, format='JPEG', quality=quality)
                buffer.seek(0)
                
                # 重新加载
                img_degraded = Image.open(buffer)
                img_tensor_degraded = torch.from_numpy(np.array(img_degraded)).float() / 255.0
                img_tensor_degraded = img_tensor_degraded.permute(2, 0, 1)
                
                degraded.append(img_tensor_degraded)
            
            return torch.stack(degraded)
        
        return degrade_op
        
    elif mode == "noise":
        std = kwargs.get("std", 0.1)
        def degrade_op(images):
            """添加高斯噪声"""
            # 确保输入是4D tensor [B, C, H, W]
            if images.dim() == 3:
                images = images.unsqueeze(0)
            
            # 生成噪声
            noise = torch.randn_like(images) * std
            
            # 添加噪声并裁剪到[0,1]范围
            degraded = images + noise
            degraded = torch.clamp(degraded, 0, 1)
            
            return degraded
        
        return degrade_op
            
    elif mode == "motion_blur":
        kernel_size = kwargs.get("kernel_size", 15)
        angle = kwargs.get("angle", 0)
        def degrade_op(images):
            """运动模糊"""
            import torch.nn.functional as F
            
            # 确保输入是4D tensor [B, C, H, W]
            if images.dim() == 3:
                images = images.unsqueeze(0)
            
            # 创建运动模糊核
            kernel = torch.zeros(kernel_size, kernel_size)
            center = kernel_size // 2
            
            # 简单的线性运动模糊
            for i in range(kernel_size):
                kernel[center, i] = 1.0 / kernel_size
            
            # 旋转核
            if angle != 0:
                # 这里简化处理，实际应该实现真正的旋转
                pass
            
            # 确保卷积核在正确的设备上
            device = images.device
            kernel = kernel.to(device)
            
            kernel = kernel.view(1, 1, kernel_size, kernel_size)
            kernel = kernel.repeat(3, 1, 1, 1)  # 扩展到3通道
            
            blurred = F.conv2d(images, kernel, padding=center, groups=3)
            return blurred
        
        return degrade_op
            
    else:
        raise ValueError(f"不支持的降质模式: {mode}")


def set_seed(seed: int):
    """设置随机种子"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class AverageMeter:
    """计算和存储平均值和当前值"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_ssl_device(device_name: str = "cuda"):
    """获取SSL训练设备"""
    if device_name.lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def save_ssl_checkpoint(state: dict, is_best: bool, save_dir: str, filename: str = "checkpoint.pth", best_filename: str = "best.pth"):
    """保存SSL训练检查点"""
    import os
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存可恢复的检查点
    checkpoint_path = os.path.join(save_dir, filename)
    torch.save(state, checkpoint_path)
    
    # 如果是最佳模型，复制为best.pth
    if is_best:
        best_path = os.path.join(save_dir, best_filename)
        import shutil
        shutil.copy(checkpoint_path, best_path)

def test_degraders():
    """测试降质器是否正常工作"""
    import torch
    
    print("测试降质器...")
    
    # 创建一个测试图像
    test_image = torch.randn(3, 64, 64)  # 3通道，64x64
    test_image = torch.clamp(test_image, 0, 1)  # 限制在0-1范围
    
    print(f"测试图像形状: {test_image.shape}")
    print(f"测试图像范围: [{test_image.min():.3f}, {test_image.max():.3f}]")
    print(f"测试图像均值: {test_image.mean():.3f}")
    print(f"测试图像方差: {test_image.var():.3f}")
    
    # 测试高斯模糊
    print("\n测试高斯模糊降质器...")
    gaussian_degrader = build_degrader("gaussian_blur", ksize=7, sigma=1.5)
    degraded_gaussian = gaussian_degrader(test_image.unsqueeze(0)).squeeze(0)
    
    print(f"高斯模糊后范围: [{degraded_gaussian.min():.3f}, {degraded_gaussian.max():.3f}]")
    print(f"高斯模糊后均值: {degraded_gaussian.mean():.3f}")
    print(f"高斯模糊后方差: {degraded_gaussian.var():.3f}")
    print(f"高斯模糊差异: 均值={abs(test_image.mean() - degraded_gaussian.mean()):.3f}, 方差={abs(test_image.var() - degraded_gaussian.var()):.3f}")
    
    # 测试双三次缩放
    print("\n测试双三次缩放降质器...")
    bicubic_degrader = build_degrader("bicubic_downup", scale=0.4)
    degraded_bicubic = bicubic_degrader(test_image.unsqueeze(0)).squeeze(0)
    
    print(f"双三次缩放后范围: [{degraded_bicubic.min():.3f}, {degraded_bicubic.max():.3f}]")
    print(f"双三次缩放后均值: {degraded_bicubic.mean():.3f}")
    print(f"双三次缩放后方差: {degraded_bicubic.var():.3f}")
    print(f"双三次缩放差异: 均值={abs(test_image.mean() - degraded_bicubic.mean()):.3f}, 方差={abs(test_image.var() - degraded_bicubic.var()):.3f}")
    
    print("\n降质器测试完成！")


# ===========================================
# 清晰度平均化工具函数
# ===========================================

def histogram_matching(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    直方图匹配，将source图像的直方图匹配到reference图像
    
    Args:
        source: 源图像 (H, W) 或 (H, W, C)
        reference: 参考图像 (H, W) 或 (H, W, C)
    
    Returns:
        匹配后的图像
    """
    if source.ndim == 3:
        # 彩色图像，分别处理每个通道
        result = np.zeros_like(source)
        for i in range(source.shape[2]):
            result[:, :, i] = histogram_matching(source[:, :, i], reference[:, :, i])
        return result
    
    # 灰度图像
    source_hist, source_bins = np.histogram(source.flatten(), bins=256, range=[0, 1])
    reference_hist, reference_bins = np.histogram(reference.flatten(), bins=256, range=[0, 1])
    
    # 计算累积分布函数
    source_cdf = source_hist.cumsum()
    reference_cdf = reference_hist.cumsum()
    
    # 归一化CDF
    source_cdf = source_cdf / source_cdf[-1]
    reference_cdf = reference_cdf / reference_cdf[-1]
    
    # 创建查找表
    lookup_table = np.zeros(256)
    for i in range(256):
        # 找到最接近的CDF值
        diff = np.abs(source_cdf[i] - reference_cdf)
        lookup_table[i] = np.argmin(diff)
    
    # 应用查找表
    source_normalized = (source * 255).astype(np.uint8)
    result = lookup_table[source_normalized].astype(np.float32) / 255.0
    
    return result


def adaptive_equalization(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    自适应直方图均衡化
    
    Args:
        image: 输入图像 (H, W) 或 (H, W, C)
        clip_limit: 对比度限制
        tile_grid_size: 网格大小
    
    Returns:
        均衡化后的图像
    """
    if image.ndim == 3:
        # 彩色图像，转换为LAB色彩空间处理
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # 只对L通道进行CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        lab[:, :, 0] = clahe.apply((lab[:, :, 0] * 255).astype(np.uint8)) / 255.0
        
        # 转换回RGB
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return result
    else:
        # 灰度图像
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        result = clahe.apply((image * 255).astype(np.uint8)) / 255.0
        return result


def contrast_stretching(image: np.ndarray, low_percent: float = 1.0, high_percent: float = 99.0) -> np.ndarray:
    """
    对比度拉伸
    
    Args:
        image: 输入图像 (H, W) 或 (H, W, C)
        low_percent: 低百分位数
        high_percent: 高百分位数
    
    Returns:
        拉伸后的图像
    """
    if image.ndim == 3:
        # 彩色图像，分别处理每个通道
        result = np.zeros_like(image)
        for i in range(image.shape[2]):
            result[:, :, i] = contrast_stretching(image[:, :, i], low_percent, high_percent)
        return result
    
    # 灰度图像
    # 计算百分位数
    low_val = np.percentile(image, low_percent)
    high_val = np.percentile(image, high_percent)
    
    # 拉伸对比度
    if high_val > low_val:
        result = (image - low_val) / (high_val - low_val)
        result = np.clip(result, 0, 1)
    else:
        result = image
    
    return result


def unsharp_masking(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0, amount: float = 1.0, threshold: float = 0) -> np.ndarray:
    """
    非锐化掩蔽，增强图像细节
    
    Args:
        image: 输入图像 (H, W) 或 (H, W, C)
        kernel_size: 高斯核大小
        sigma: 高斯核标准差
        amount: 锐化强度
        threshold: 阈值
    
    Returns:
        锐化后的图像
    """
    if image.ndim == 3:
        # 彩色图像，分别处理每个通道
        result = np.zeros_like(image)
        for i in range(image.shape[2]):
            result[:, :, i] = unsharp_masking(image[:, :, i], kernel_size, sigma, amount, threshold)
        return result
    
    # 灰度图像
    # 高斯模糊
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    # 计算锐化掩蔽
    mask = image - blurred
    
    # 应用锐化
    if threshold > 0:
        mask = np.where(np.abs(mask) > threshold, mask, 0)
    
    result = image + amount * mask
    result = np.clip(result, 0, 1)
    
    return result


def apply_sharpness_averaging(image: np.ndarray, target_sharpness: float, method: str = "histogram_matching", 
                             reference_image: np.ndarray = None, **kwargs) -> np.ndarray:
    """
    应用清晰度平均化
    
    Args:
        image: 输入图像
        target_sharpness: 目标清晰度
        method: 平均化方法
        reference_image: 参考图像（用于直方图匹配）
        **kwargs: 其他参数
    
    Returns:
        平均化后的图像
    """
    if method == "histogram_matching":
        if reference_image is None:
            raise ValueError("直方图匹配方法需要提供参考图像")
        return histogram_matching(image, reference_image)
    
    elif method == "adaptive_equalization":
        clip_limit = kwargs.get("clip_limit", 2.0)
        tile_grid_size = kwargs.get("tile_grid_size", (8, 8))
        return adaptive_equalization(image, clip_limit, tile_grid_size)
    
    elif method == "contrast_stretching":
        low_percent = kwargs.get("low_percent", 1.0)
        high_percent = kwargs.get("high_percent", 99.0)
        return contrast_stretching(image, low_percent, high_percent)
    
    elif method == "unsharp_masking":
        kernel_size = kwargs.get("kernel_size", 5)
        sigma = kwargs.get("sigma", 1.0)
        amount = kwargs.get("amount", 1.0)
        threshold = kwargs.get("threshold", 0)
        return unsharp_masking(image, kernel_size, sigma, amount, threshold)
    
    else:
        raise ValueError(f"不支持的清晰度平均化方法: {method}")


def calculate_target_sharpness(sharpness_data: Dict[str, Dict[str, float]], target_type: str = "mean", 
                              custom_value: float = None) -> float:
    """
    计算目标清晰度值
    
    Args:
        sharpness_data: 清晰度数据字典
        target_type: 目标类型 ("mean", "median", "custom")
        custom_value: 自定义值
    
    Returns:
        目标清晰度值
    """
    if target_type == "custom" and custom_value is not None:
        return custom_value
    
    # 提取所有图像的清晰度分数
    sharpness_scores = []
    for img_path, metrics in sharpness_data.items():
        if "sharpness" in metrics:
            sharpness_scores.append(metrics["sharpness"])
    
    if not sharpness_scores:
        return 0.5  # 默认值
    
    if target_type == "mean":
        return float(np.mean(sharpness_scores))
    elif target_type == "median":
        return float(np.median(sharpness_scores))
    else:
        return float(np.mean(sharpness_scores))


def find_reference_image(sharpness_data: Dict[str, Dict[str, float]], target_sharpness: float) -> str:
    """
    找到最接近目标清晰度的参考图像
    
    Args:
        sharpness_data: 清晰度数据字典
        target_sharpness: 目标清晰度
    
    Returns:
        参考图像路径
    """
    best_match = None
    min_diff = float('inf')
    
    for img_path, metrics in sharpness_data.items():
        if "sharpness" in metrics:
            diff = abs(metrics["sharpness"] - target_sharpness)
            if diff < min_diff:
                min_diff = diff
                best_match = img_path
    
    return best_match


def save_image_to_dataset(image: np.ndarray, original_path: str, dataset_name: str, 
                         output_dir: Path) -> Path:
    """
    将图像保存到指定数据集目录
    
    Args:
        image: 图像数据
        original_path: 原始图像路径
        suffix: 后缀名
        dataset_name: 数据集名称
        output_dir: 输出目录
    
    Returns:
        保存后的图像路径
    """
    # 获取原始文件名
    original_name = Path(original_path).name
    output_path = output_dir / original_name
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存图像
    if len(image.shape) == 3:
        # 彩色图像，转换为BGR格式保存
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), (image_bgr * 255).astype(np.uint8))
    else:
        # 灰度图像
        cv2.imwrite(str(output_path), (image * 255).astype(np.uint8))
    
    return output_path


if __name__ == "__main__":
    # 运行测试
    test_degraders()


# ===========================================
# 清晰度平均化工具函数
# ===========================================

def histogram_matching(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    直方图匹配，将source图像的直方图匹配到reference图像
    
    Args:
        source: 源图像 (H, W) 或 (H, W, C)
        reference: 参考图像 (H, W) 或 (H, W, C)
    
    Returns:
        匹配后的图像
    """
    if source.ndim == 3:
        # 彩色图像，分别处理每个通道
        result = np.zeros_like(source)
        for i in range(source.shape[2]):
            result[:, :, i] = histogram_matching(source[:, :, i], reference[:, :, i])
        return result
    
    # 灰度图像
    source_hist, source_bins = np.histogram(source.flatten(), bins=256, range=[0, 1])
    reference_hist, reference_bins = np.histogram(reference.flatten(), bins=256, range=[0, 1])
    
    # 计算累积分布函数
    source_cdf = source_hist.cumsum()
    reference_cdf = reference_hist.cumsum()
    
    # 归一化CDF
    source_cdf = source_cdf / source_cdf[-1]
    reference_cdf = reference_cdf / reference_cdf[-1]
    
    # 创建查找表
    lookup_table = np.zeros(256)
    for i in range(256):
        # 找到最接近的CDF值
        diff = np.abs(source_cdf[i] - reference_cdf)
        lookup_table[i] = np.argmin(diff)
    
    # 应用查找表
    source_normalized = (source * 255).astype(np.uint8)
    result = lookup_table[source_normalized].astype(np.float32) / 255.0
    
    return result


def adaptive_equalization(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    自适应直方图均衡化
    
    Args:
        image: 输入图像 (H, W) 或 (H, W, C)
        clip_limit: 对比度限制
        tile_grid_size: 网格大小
    
    Returns:
        均衡化后的图像
    """
    if image.ndim == 3:
        # 彩色图像，转换为LAB色彩空间处理
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # 只对L通道进行CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        lab[:, :, 0] = clahe.apply((lab[:, :, 0] * 255).astype(np.uint8)) / 255.0
        
        # 转换回RGB
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return result
    else:
        # 灰度图像
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        result = clahe.apply((image * 255).astype(np.uint8)) / 255.0
        return result


def contrast_stretching(image: np.ndarray, low_percent: float = 1.0, high_percent: float = 99.0) -> np.ndarray:
    """
    对比度拉伸
    
    Args:
        image: 输入图像 (H, W) 或 (H, W, C)
        low_percent: 低百分位数
        high_percent: 高百分位数
    
    Returns:
        拉伸后的图像
    """
    if image.ndim == 3:
        # 彩色图像，分别处理每个通道
        result = np.zeros_like(image)
        for i in range(image.shape[2]):
            result[:, :, i] = contrast_stretching(image[:, :, i], low_percent, high_percent)
        return result
    
    # 灰度图像
    # 计算百分位数
    low_val = np.percentile(image, low_percent)
    high_val = np.percentile(image, high_percent)
    
    # 拉伸对比度
    if high_val > low_val:
        result = (image - low_val) / (high_val - low_val)
        result = np.clip(result, 0, 1)
    else:
        result = image
    
    return result


def unsharp_masking(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0, amount: float = 1.0, threshold: float = 0) -> np.ndarray:
    """
    非锐化掩蔽，增强图像细节
    
    Args:
        image: 输入图像 (H, W) 或 (H, W, C)
        kernel_size: kernel_size: 高斯核大小
        sigma: 高斯核标准差
        amount: 锐化强度
        threshold: 阈值
    
    Returns:
        锐化后的图像
    """
    if image.ndim == 3:
        # 彩色图像，分别处理每个通道
        result = np.zeros_like(image)
        for i in range(image.shape[2]):
            result[:, :, i] = unsharp_masking(image[:, :, i], kernel_size, sigma, amount, threshold)
        return result
    
    # 灰度图像
    # 高斯模糊
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    # 计算锐化掩蔽
    mask = image - blurred
    
    # 应用锐化
    if threshold > 0:
        mask = np.where(np.abs(mask) > threshold, mask, 0)
    
    result = image + amount * mask
    result = np.clip(result, 0, 1)
    
    return result


def apply_sharpness_averaging(image: np.ndarray, target_sharpness: float, method: str = "histogram_matching", 
                             reference_image: np.ndarray = None, **kwargs) -> np.ndarray:
    """
    应用清晰度平均化
    
    Args:
        image: 输入图像
        target_sharpness: 目标清晰度
        method: 平均化方法
        reference_image: 参考图像（用于直方图匹配）
        **kwargs: 其他参数
    
    Returns:
        平均化后的图像
    """
    if method == "histogram_matching":
        if reference_image is None:
            raise ValueError("直方图匹配方法需要提供参考图像")
        return histogram_matching(image, reference_image)
    
    elif method == "adaptive_equalization":
        clip_limit = kwargs.get("clip_limit", 2.0)
        tile_grid_size = kwargs.get("tile_grid_size", (8, 8))
        return adaptive_equalization(image, clip_limit, tile_grid_size)
    
    elif method == "contrast_stretching":
        low_percent = kwargs.get("low_percent", 1.0)
        high_percent = kwargs.get("high_percent", 99.0)
        return contrast_stretching(image, low_percent, high_percent)
    
    elif method == "unsharp_masking":
        kernel_size = kwargs.get("kernel_size", 5)
        sigma = kwargs.get("sigma", 1.0)
        amount = kwargs.get("amount", 1.0)
        threshold = kwargs.get("threshold", 0)
        return unsharp_masking(image, kernel_size, sigma, amount, threshold)
    
    else:
        raise ValueError(f"不支持的清晰度平均化方法: {method}")


def calculate_target_sharpness(sharpness_data: Dict[str, Dict[str, float]], target_type: str = "mean", 
                              custom_value: float = None) -> float:
    """
    计算目标清晰度值
    
    Args:
        sharpness_data: 清晰度数据字典
        target_type: 目标类型 ("mean", "median", "custom")
        custom_value: 自定义值
    
    Returns:
        目标清晰度值
    """
    if target_type == "custom" and custom_value is not None:
        return custom_value
    
    # 提取所有图像的清晰度分数
    sharpness_scores = []
    for img_path, metrics in sharpness_data.items():
        if "sharpness" in metrics:
            sharpness_scores.append(metrics["sharpness"])
    
    if not sharpness_scores:
        return 0.5  # 默认值
    
    if target_type == "mean":
        return float(np.mean(sharpness_scores))
    elif target_type == "median":
        return float(np.median(sharpness_scores))
    else:
        return float(np.mean(sharpness_scores))


def find_reference_image(sharpness_data: Dict[str, Dict[str, float]], target_sharpness: float) -> str:
    """
    找到最接近目标清晰度的参考图像
    
    Args:
        sharpness_data: 清晰度数据字典
        target_sharpness: 目标清晰度
    
    Returns:
        参考图像路径
    """
    best_match = None
    min_diff = float('inf')
    
    for img_path, metrics in sharpness_data.items():
        if "sharpness" in metrics:
            diff = abs(metrics["sharpness"] - target_sharpness)
            if diff < min_diff:
                min_diff = diff
                best_match = img_path
    
    return best_match


def save_image_to_dataset(image: np.ndarray, original_path: str, dataset_name: str, 
                         output_dir: Path) -> Path:
    """
    将图像保存到指定数据集目录
    
    Args:
        image: 图像数据
        original_path: 原始图像路径
        dataset_name: 数据集名称
        output_dir: 输出目录
    
    Returns:
        保存后的图像路径
    """
    # 获取原始文件名
    original_name = Path(original_path).name
    output_path = output_dir / original_name
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存图像
    if len(image.shape) == 3:
        # 彩色图像，转换为BGR格式保存
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), (image_bgr * 255).astype(np.uint8))
    else:
        # 灰度图像
        cv2.imwrite(str(output_path), (image * 255).astype(np.uint8))
    
    return output_path
