# utils/smart_annotation_util.py
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, Slider
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import json
import os

class SmartAnnotationTool:
    """智能标注工具，提供多种提高标注效率的功能"""
    
    def __init__(self):
        self.current_image = None
        self.current_mask = None
        self.prediction_mask = None
        self.annotation_history = []
        self.template_masks = {}
        
    def load_image_and_prediction(self, image_path: str, prediction_path: str):
        """加载图像和预测mask"""
        self.current_image = cv2.imread(image_path)
        self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        
        if os.path.exists(prediction_path):
            self.prediction_mask = cv2.imread(prediction_path)
            self.prediction_mask = cv2.cvtColor(self.prediction_mask, cv2.COLOR_BGR2RGB)
            self.current_mask = self.prediction_mask.copy()
        else:
            self.current_mask = np.zeros_like(self.current_image)
    
    def create_interactive_interface(self):
        """创建交互式标注界面"""
        self.root = tk.Tk()
        self.root.title("智能标注工具")
        self.root.geometry("1400x900")
        
        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧控制面板
        self._create_control_panel(main_frame)
        
        # 右侧图像显示区域
        self._create_image_display(main_frame)
        
        # 绑定快捷键
        self._bind_shortcuts()
        
    def _create_control_panel(self, parent):
        """创建控制面板"""
        control_frame = ttk.LabelFrame(parent, text="标注控制", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # 类别选择
        ttk.Label(control_frame, text="选择类别:").pack(anchor=tk.W)
        self.class_var = tk.StringVar(value="carbon")
        class_combo = ttk.Combobox(control_frame, textvariable=self.class_var, 
                                  values=["carbon", "SE", "AM"], state="readonly")
        class_combo.pack(fill=tk.X, pady=(0, 10))
        class_combo.bind("<<ComboboxSelected>>", self._on_class_change)
        
        # 工具选择
        ttk.Label(control_frame, text="标注工具:").pack(anchor=tk.W)
        self.tool_var = tk.StringVar(value="brush")
        tools = [("画笔", "brush"), ("橡皮擦", "eraser"), ("魔棒", "magic_wand"), 
                ("多边形", "polygon"), ("圆形", "circle")]
        
        for text, value in tools:
            ttk.Radiobutton(control_frame, text=text, variable=self.tool_var, 
                          value=value).pack(anchor=tk.W)
        
        # 画笔大小
        ttk.Label(control_frame, text="画笔大小:").pack(anchor=tk.W)
        self.brush_size_var = tk.IntVar(value=10)
        brush_scale = ttk.Scale(control_frame, from_=1, to=50, variable=self.brush_size_var, 
                               orient=tk.HORIZONTAL)
        brush_scale.pack(fill=tk.X, pady=(0, 10))
        
        # 智能功能
        ttk.Label(control_frame, text="智能功能:").pack(anchor=tk.W)
        self.auto_refine_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="自动边界细化", 
                       variable=self.auto_refine_var).pack(anchor=tk.W)
        
        self.template_match_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="模板匹配", 
                       variable=self.template_match_var).pack(anchor=tk.W)
        
        # 操作按钮
        ttk.Button(control_frame, text="应用预测", 
                  command=self._apply_prediction).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="智能修正", 
                  command=self._smart_correction).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="边界细化", 
                  command=self._refine_boundaries).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="撤销", 
                  command=self._undo).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="重做", 
                  command=self._redo).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="保存", 
                  command=self._save_annotation).pack(fill=tk.X, pady=2)
        
        # 状态显示
        self.status_label = ttk.Label(control_frame, text="就绪")
        self.status_label.pack(anchor=tk.W, pady=(10, 0))
    
    def _create_image_display(self, parent):
        """创建图像显示区域"""
        image_frame = ttk.Frame(parent)
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 创建画布
        self.canvas = tk.Canvas(image_frame, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 绑定鼠标事件
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.canvas.bind("<Motion>", self._on_canvas_motion)
        
        # 初始化绘图状态
        self.drawing = False
        self.last_x = None
        self.last_y = None
    
    def _bind_shortcuts(self):
        """绑定快捷键"""
        self.root.bind("<Control-z>", lambda e: self._undo())
        self.root.bind("<Control-y>", lambda e: self._redo())
        self.root.bind("<Control-s>", lambda e: self._save_annotation())
        self.root.bind("<Control-r>", lambda e: self._refine_boundaries())
        self.root.bind("<Control-m>", lambda e: self._smart_correction())
    
    def _on_class_change(self, event):
        """类别改变事件"""
        class_map = {"carbon": 0, "SE": 1, "AM": 2}
        self.current_class = class_map[self.class_var.get()]
        self.status_label.config(text=f"当前类别: {self.class_var.get()}")
    
    def _on_canvas_click(self, event):
        """画布点击事件"""
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y
        
        if self.tool_var.get() == "magic_wand":
            self._magic_wand_selection(event.x, event.y)
    
    def _on_canvas_drag(self, event):
        """画布拖拽事件"""
        if not self.drawing:
            return
        
        if self.tool_var.get() == "brush":
            self._draw_line(self.last_x, self.last_y, event.x, event.y)
        elif self.tool_var.get() == "eraser":
            self._erase_line(self.last_x, self.last_y, event.x, event.y)
        
        self.last_x = event.x
        self.last_y = event.y
    
    def _on_canvas_release(self, event):
        """画布释放事件"""
        self.drawing = False
        
        if self.auto_refine_var.get():
            self._auto_refine_boundaries()
    
    def _on_canvas_motion(self, event):
        """画布移动事件"""
        self.status_label.config(text=f"位置: ({event.x}, {event.y})")
    
    def _draw_line(self, x1, y1, x2, y2):
        """绘制线条"""
        # 这里需要实现实际的绘制逻辑
        # 将绘制操作应用到self.current_mask
        pass
    
    def _erase_line(self, x1, y1, x2, y2):
        """擦除线条"""
        # 这里需要实现实际的擦除逻辑
        pass
    
    def _magic_wand_selection(self, x, y):
        """魔棒选择"""
        # 实现基于颜色相似性的区域选择
        pass
    
    def _apply_prediction(self):
        """应用预测结果"""
        if self.prediction_mask is not None:
            self.current_mask = self.prediction_mask.copy()
            self._update_display()
            self.status_label.config(text="已应用预测结果")
    
    def _smart_correction(self):
        """智能修正"""
        # 基于模板匹配和模式识别的智能修正
        if self.template_match_var.get():
            self._apply_template_matching()
        
        self.status_label.config(text="智能修正完成")
    
    def _apply_template_matching(self):
        """应用模板匹配"""
        # 实现模板匹配逻辑
        pass
    
    def _refine_boundaries(self):
        """边界细化"""
        # 使用形态学操作细化边界
        if self.current_mask is not None:
            kernel = np.ones((3, 3), np.uint8)
            refined_mask = cv2.morphologyEx(self.current_mask, cv2.MORPH_CLOSE, kernel)
            self.current_mask = refined_mask
            self._update_display()
            self.status_label.config(text="边界细化完成")
    
    def _auto_refine_boundaries(self):
        """自动边界细化"""
        if self.auto_refine_var.get():
            self._refine_boundaries()
    
    def _undo(self):
        """撤销操作"""
        if self.annotation_history:
            self.current_mask = self.annotation_history.pop()
            self._update_display()
            self.status_label.config(text="已撤销")
    
    def _redo(self):
        """重做操作"""
        # 实现重做逻辑
        pass
    
    def _save_annotation(self):
        """保存标注结果"""
        if self.current_mask is not None:
            from tkinter import filedialog
            output_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
            )
            if output_path:
                mask_bgr = cv2.cvtColor(self.current_mask, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, mask_bgr)
                self.status_label.config(text=f"已保存到: {output_path}")
    
    def _update_display(self):
        """更新显示"""
        # 更新画布显示
        pass
    
    def run(self):
        """运行标注工具"""
        if hasattr(self, 'root'):
            self.root.mainloop()


class BatchAnnotationProcessor:
    """批量标注处理器"""
    
    def __init__(self):
        self.annotation_queue = []
        self.processed_count = 0
        
    def add_to_queue(self, image_path: str, prediction_path: str):
        """添加到处理队列"""
        self.annotation_queue.append({
            'image_path': image_path,
            'prediction_path': prediction_path,
            'status': 'pending'
        })
    
    def process_batch(self, batch_size: int = 5):
        """批量处理标注"""
        for i in range(0, len(self.annotation_queue), batch_size):
            batch = self.annotation_queue[i:i+batch_size]
            self._process_batch(batch)
    
    def _process_batch(self, batch):
        """处理单个批次"""
        for item in batch:
            try:
                # 创建标注工具实例
                tool = SmartAnnotationTool()
                tool.load_image_and_prediction(item['image_path'], item['prediction_path'])
                
                # 这里可以添加批量处理逻辑
                item['status'] = 'processed'
                self.processed_count += 1
                
            except Exception as e:
                print(f"处理 {item['image_path']} 时出错: {e}")
                item['status'] = 'error'


def create_smart_annotation_tool():
    """创建智能标注工具"""
    return SmartAnnotationTool()


def create_batch_processor():
    """创建批量处理器"""
    return BatchAnnotationProcessor()


# 使用示例
if __name__ == "__main__":
    # 创建智能标注工具
    tool = create_smart_annotation_tool()
    
    # 加载示例数据
    image_path = "data/labeled/4.2V-009.png"
    prediction_path = "active_learning/predictions/iteration_1/4.2V-009_prediction_mask.png"
    
    if os.path.exists(image_path):
        tool.load_image_and_prediction(image_path, prediction_path)
        tool.create_interactive_interface()
        tool.run()
    else:
        print("示例文件不存在") 