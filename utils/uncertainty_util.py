# utils/uncertainty_util.py
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Tuple

def compute_entropy(probs):
    """计算熵不确定性"""
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
    return entropy.mean(dim=[1,2])

def compute_margin(probs):
    """计算边缘不确定性"""
    top2 = torch.topk(probs, 2, dim=1)[0]
    margin = top2[:, 0] - top2[:, 1]
    return (1 - margin).mean(dim=[1,2])

def compute_least_confidence(probs):
    """计算最小置信度不确定性"""
    max_prob = torch.max(probs, dim=1)[0]
    return (1 - max_prob).mean(dim=[1,2])

def detect_prediction_boundaries(prediction_mask: np.ndarray, dilation_size: int = 3) -> np.ndarray:
    """检测预测掩码的边界"""
    boundaries = np.zeros_like(prediction_mask, dtype=bool)
    
    for class_id in np.unique(prediction_mask):
        if class_id == 0:  # 跳过背景
            continue
            
        class_mask = (prediction_mask == class_id).astype(np.uint8)
        
        # 形态学操作检测边界
        kernel = np.ones((dilation_size, dilation_size), np.uint8)
        eroded = cv2.erode(class_mask, kernel, iterations=1)
        dilated = cv2.dilate(class_mask, kernel, iterations=1)
        boundary = dilated - eroded
        
        boundaries = boundaries | (boundary > 0)
    
    return boundaries

def compute_boundary_entropy(probs: torch.Tensor, prediction_mask: torch.Tensor, 
                           dilation_size: int = 3) -> torch.Tensor:
    """计算边界区域的熵不确定性"""
    batch_size = probs.shape[0]
    boundary_uncertainties = []
    
    for b in range(batch_size):
        # 检测边界
        pred_mask_np = prediction_mask[b].cpu().numpy()
        boundaries = detect_prediction_boundaries(pred_mask_np, dilation_size)
        
        if not np.any(boundaries):
            boundary_uncertainties.append(torch.tensor(0.0, device=probs.device))
            continue
        
        # 只在边界区域计算熵
        boundary_probs = probs[b, :, boundaries]
        if boundary_probs.numel() == 0:
            boundary_uncertainties.append(torch.tensor(0.0, device=probs.device))
            continue
            
        entropy = -torch.sum(boundary_probs * torch.log(boundary_probs + 1e-10), dim=0)
        boundary_uncertainties.append(entropy.mean())
    
    return torch.stack(boundary_uncertainties)

def compute_boundary_margin(probs: torch.Tensor, prediction_mask: torch.Tensor,
                          dilation_size: int = 3) -> torch.Tensor:
    """计算边界区域的边缘不确定性"""
    batch_size = probs.shape[0]
    boundary_uncertainties = []
    
    for b in range(batch_size):
        # 检测边界
        pred_mask_np = prediction_mask[b].cpu().numpy()
        boundaries = detect_prediction_boundaries(pred_mask_np, dilation_size)
        
        if not np.any(boundaries):
            boundary_uncertainties.append(torch.tensor(0.0, device=probs.device))
            continue
        
        # 只在边界区域计算边缘不确定性
        boundary_probs = probs[b, :, boundaries]
        if boundary_probs.numel() == 0:
            boundary_uncertainties.append(torch.tensor(0.0, device=probs.device))
            continue
            
        # 计算前两个最大概率的差值
        top2 = torch.topk(boundary_probs, 2, dim=0)[0]
        margin = top2[0] - top2[1]
        boundary_uncertainties.append((1 - margin).mean())
    
    return torch.stack(boundary_uncertainties)

def compute_gradient_uncertainty(logits: torch.Tensor, prediction_mask: torch.Tensor) -> torch.Tensor:
    """基于梯度的边界不确定性"""
    batch_size = logits.shape[0]
    gradient_uncertainties = []
    
    for b in range(batch_size):
        # 计算logits的梯度
        logits_b = logits[b].unsqueeze(0)  # (1, C, H, W)
        
        # 计算空间梯度
        grad_x = torch.gradient(logits_b, dim=3)[0]  # 水平梯度
        grad_y = torch.gradient(logits_b, dim=2)[0]  # 垂直梯度
        
        # 计算梯度幅值
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        
        # 在预测边界区域计算不确定性
        pred_mask_np = prediction_mask[b].cpu().numpy()
        boundaries = detect_prediction_boundaries(pred_mask_np)
        
        if not np.any(boundaries):
            gradient_uncertainties.append(torch.tensor(0.0, device=logits.device))
            continue
        
        # 在边界区域计算平均梯度幅值
        boundary_gradients = gradient_magnitude[0, :, boundaries]
        if boundary_gradients.numel() == 0:
            gradient_uncertainties.append(torch.tensor(0.0, device=logits.device))
            continue
            
        gradient_uncertainties.append(boundary_gradients.mean())
    
    return torch.stack(gradient_uncertainties)

def get_uncertainty_function(method: str):
    """获取不确定性计算函数"""
    if method == 'entropy':
        return compute_entropy
    elif method == 'margin':
        return compute_margin
    elif method == 'least_confidence':
        return compute_least_confidence
    elif method == 'boundary_entropy':
        return compute_boundary_entropy
    elif method == 'boundary_margin':
        return compute_boundary_margin
    elif method == 'gradient_uncertainty':
        return compute_gradient_uncertainty
    else:
        raise ValueError(f"未知的不确定性方法: {method}")

def compute_combined_uncertainty(logits: torch.Tensor, methods: List[str], 
                               prediction_mask: torch.Tensor = None,
                               dilation_size: int = 3) -> torch.Tensor:
    """计算组合不确定性"""
    probs = F.softmax(logits, dim=1)
    uncertainties = []
    
    for method in methods:
        if method in ['boundary_entropy', 'boundary_margin']:
            # 边界不确定性方法需要预测掩码
            if prediction_mask is None:
                prediction_mask = torch.argmax(logits, dim=1)
            
            if method == 'boundary_entropy':
                unc = compute_boundary_entropy(probs, prediction_mask, dilation_size)
            elif method == 'boundary_margin':
                unc = compute_boundary_margin(probs, prediction_mask, dilation_size)
        elif method == 'gradient_uncertainty':
            # 梯度不确定性需要预测掩码
            if prediction_mask is None:
                prediction_mask = torch.argmax(logits, dim=1)
            unc = compute_gradient_uncertainty(logits, prediction_mask)
        else:
            # 传统不确定性方法
            unc_func = get_uncertainty_function(method)
            unc = unc_func(probs)
        
        uncertainties.append(unc)
    
    combined = torch.stack(uncertainties).mean(dim=0)
    return combined 