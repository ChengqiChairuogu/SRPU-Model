# SRPU-Model: 电极SEM表征图象智能分割平台

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2](https://img.shields.io/badge/pytorch-2.2.0-red.svg)](https://pytorch.org/)

## 项目简介

**SRPU-Model** 是一个专为扫描电镜（SEM）图像分割设计的自进化深度学习平台。该平台集成了多种先进的机器学习策略，包括**监督学习**、**自监督学习**、**半监督学习**和**主动学习**，旨在通过有限标注数据和大量无标注数据，持续提升分割模型性能，实现"自我成长"的智能分割系统。

### 核心特性

- **专业领域**: 专为SEM图像分割优化，支持多类别分割（AM、SE、carbon）
- **自进化学习**: 集成主动学习，智能选择最有价值的样本进行标注
- **多策略融合**: 支持监督、自监督、半监督学习策略的灵活组合
- **全面评估**: 提供定性和定量评估，支持可视化分析
- **高效训练**: 支持多阶段训练和微调，优化训练效率
- **智能标注**: 自动生成预测mask，辅助人工标注

### 技术架构

```
SRPU-Model/
├── configs/          # 配置文件管理
├── data/             # 数据集存储
├── datasets/         # PyTorch数据集定义
├── models/           # 模型架构与权重
├── tasks/            # 训练/推理/评估脚本
├── pipelines/        # 多阶段训练管道
├── utils/            # 工具函数库
├── json/             # 数据集索引文件
└── active_learning/  # 主动学习输出
```

---

## 快速开始

### 1. 环境配置

#### 使用Conda（推荐）
```bash
# 克隆项目
git clone <repository-url>
cd SRPU-Model

# 创建并激活环境
conda env create -f environment.yml
conda activate SRPU-Model
```

#### 使用pip
```bash
# 安装基础依赖
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0
pip install numpy==1.26 pandas scikit-image opencv matplotlib
pip install albumentations tqdm wandb torchmetrics pillow scikit-learn
```

### 2. 数据准备

#### 数据目录结构
```
data/
├── raw/                 # 原始SEM图像
│   ├── 4.2V-001.png
│   ├── 4.2V-002.png
│   └── ...
├── labeled/             # 已标注图像和掩码
│   ├── 4.2V-009.png    # 原始图像
│   ├── 4.2V-009_mask.png  # 对应掩码
│   └── ...
└── inference/
    ├── input/           # 待推理图像
    └── output/          # 推理结果
```

#### 生成数据集索引
```bash
# 自动生成所有数据集索引
python json_generator.py --mode generate_all
```

### 3. 基础训练

#### 监督训练
```bash
# 单阶段监督训练
python tasks/train_task.py

# 多阶段监督训练
python pipelines/multistage_train_pipeline.py
```

#### 自监督预训练
```bash
# 使用MAE进行自监督预训练
python tasks/ssl_pretrain_task.py
```

#### 模型微调
```bash
# 单阶段微调
python tasks/finetune_task.py

# 多阶段微调
python pipelines/multistage_finetune_pipeline.py
```

### 4. 模型评估

```bash
# 统一评估脚本
python tasks/evaluation_task.py
```

### 5. 模型推理

```bash
# 将待分割图像放入 data/inference/input/
python tasks/inference_task.py
# 结果保存在 data/inference/output/
```

---

## 主动学习

### 概述
主动学习模块通过智能选择最有价值的样本进行人工标注，最大化标注效率，减少标注成本。

### 核心功能

#### 1. 多种不确定性策略
- **熵不确定性**: 基于预测概率分布的熵
- **置信度不确定性**: 基于预测的最大置信度  
- **边缘不确定性**: 基于最高概率与次高概率的差值
- **集成不确定性**: 基于多个模型的预测方差

#### 2. 多样性采样策略
- **核心集选择**: 确保选择的样本具有代表性
- **聚类选择**: 基于聚类的多样性采样

#### 3. 智能数据管理
- 自动备份原始数据
- 动态管理已标注和未标注数据
- 详细的选择历史记录

#### 4. 预测Mask生成
- 自动为最不确定样本生成预测mask
- 支持彩色mask输出，便于可视化
- 可直接用于手动标注调整

### 使用方法

#### 1. 配置参数
编辑 `configs/active_learning/active_learning_config.py`:

```python
# 主动学习核心参数
INITIAL_LABELED_SIZE = 50      # 初始标注样本数量
BATCH_SIZE_TO_LABEL = 20       # 每轮选择的样本数量
MAX_ACTIVE_LEARNING_ROUNDS = 10  # 最大主动学习轮数
MIN_IMPROVEMENT_THRESHOLD = 0.01  # 性能提升阈值

# 不确定性策略配置
UNCERTAINTY_STRATEGIES = {
    "entropy": {"enabled": True, "weight": 1.0},
    "confidence": {"enabled": True, "weight": 1.0},
    "margin": {"enabled": True, "weight": 1.0}
}
```

#### 2. 运行主动学习
```bash
python tasks/active_learning_task.py
```

#### 3. 工作流程
1. **初始化**: 系统自动备份原始数据，创建初始训练集
2. **训练**: 使用当前已标注数据训练模型
3. **不确定性计算**: 对未标注样本计算不确定性分数
4. **样本选择**: 基于不确定性和多样性策略选择样本
5. **人工标注**: 用户对选中的样本进行标注
6. **数据更新**: 将新标注的样本添加到训练集
7. **重复**: 重复步骤2-6直到满足停止条件

### 输出结果

#### 1. 检查点文件
```
models/checkpoints/active_learning_unet_unet/
├── round_1_best.pth          # 第1轮最佳模型
├── round_1_final.pth         # 第1轮最终模型
├── round_2_best.pth          # 第2轮最佳模型
├── ...
├── training_history.json      # 训练历史
└── final_statistics.json     # 最终统计
```

#### 2. 可视化结果
```
active_learning_results/
├── round_1/
│   ├── sample1_uncertainty.png    # 不确定性热力图
│   ├── sample2_uncertainty.png
│   └── selection_summary.json     # 选择摘要
├── round_2/
│   └── ...
```

#### 3. 预测Mask输出
```
active_learning_predictions/
├── round_1/
│   ├── sample1_predicted_mask.png    # 预测mask
│   ├── sample2_predicted_mask.png
│   └── selection_info.json           # 选择信息
├── round_2/
│   └── ...
```

### 手动标注指南

#### 使用ImageJ进行标注
1. **下载预测文件**
```bash
scp -r username@server_ip:~/SRPU-Model/active_learning/predictions/iteration_1/ ./
```

2. **使用ImageJ调整**
   - 打开ImageJ应用程序
   - 加载预测mask图像
   - 使用画笔工具修正边界
   - 应用形态学操作平滑边缘
   - 保存修改后的mask

3. **上传标注文件**
```bash
scp image_name_mask.png username@server_ip:~/SRPU-Model/data/labeled/
scp image_name.png username@server_ip:~/SRPU-Model/data/labeled/
```

---

## 模型评估

### 评估指标
- **Dice Score**: 评估分割重叠度
- **IoU Score**: 评估交并比
- **可视化分析**: 原图、真值、预测对比

### 评估脚本
```bash
# 统一评估
python tasks/evaluation_task.py

# 自监督模型评估
python tasks/ssl_inspect_task.py
```

---

## 配置管理

### 基础配置 (`configs/base.py`)
```python
# 图像配置
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
INPUT_DEPTH = 3
NUM_CLASSES = 3

# 类别映射
CLASS_NAMES = {
    0: "carbon", 
    1: "SE", 
    2: "AM"
}

# 颜色映射
COLOR_MAPPING = {
    0: [0, 0, 255],      # carbon
    1: [0, 255, 255],    # SE - 青色
    2: [255, 126, 126]   # AM - 红色
}
```

### 训练配置
- **单阶段训练**: `configs/train/train_config.py`
- **多阶段训练**: `configs/train/multistage_train_config.py`
- **微调配置**: `configs/finetune/finetune_config.py`
- **自监督配置**: `configs/selfup/ssl_config.py`

### 日志系统
```python
# 在配置文件中选择日志系统
LOGGER = "wandb"  # 可选 "wandb" 或 "tensorboard"
```

---

## 项目结构详解

### 核心模块

#### configs/ - 配置管理
```
configs/
├── base.py                    # 全局基础配置
├── augmentation_config.py     # 数据增强参数
├── dataset_config.py          # 数据集配置
├── wandb_config.py           # 日志系统配置
├── active_learning/          # 主动学习配置
├── train/                    # 训练配置
├── finetune/                 # 微调配置
├── inference/                # 推理配置
└── selfup/                   # 自监督配置
```

#### tasks/ - 任务脚本
```
tasks/
├── train_task.py             # 监督训练
├── finetune_task.py          # 模型微调
├── ssl_pretrain_task.py      # 自监督预训练
├── active_learning_task.py   # 主动学习
├── inference_task.py         # 模型推理
├── evaluation_task.py        # 模型评估
└── ssl_inspect_task.py      # 自监督评估
```

#### models/ - 模型架构
```
models/
├── segmentation_unet.py       # U-Net分割模型
├── unet_autoencoder.py       # U-Net自编码器
├── mae_model.py              # MAE自监督模型
├── encoders/                 # 编码器模块
│   ├── efficientnet_encoder.py
│   ├── dinov2_encoder.py
│   └── ...
├── decoders/                 # 解码器模块
│   ├── unet_decoder.py
│   ├── deeplab_decoder.py
│   └── ...
└── checkpoints/              # 模型权重
```

#### utils/ - 工具函数
```
utils/
├── augmentation_util.py       # 数据增强
├── training_util.py          # 训练工具
├── uncertainty_util.py       # 不确定性计算
├── logging_util.py           # 日志管理
├── stratified_sample_util.py # 分层采样
└── ...
```

#### pipelines/ - 多阶段训练
```
pipelines/
├── multistage_train_pipeline.py      # 多阶段训练
└── multistage_finetune_pipeline.py   # 多阶段微调
```

---

## 高级特性

### 多阶段训练/微调
支持复杂的多阶段训练流程，每阶段可自定义：
- 数据集组合
- 训练轮数
- 学习率策略
- 权重加载

```bash
# 多阶段训练
python pipelines/multistage_train_pipeline.py

# 多阶段微调
python pipelines/multistage_finetune_pipeline.py
```

### 预训练模型使用
支持多种预训练模型加载策略：

```python
# 配置预训练模型
USE_PRETRAINED_MODEL = True
PRETRAINED_MODEL_PATH = Path("models/checkpoints/ssl_pretrained_model.pth")
```

### 日志与实验管理
- **Wandb**: 支持在线/离线模式
- **Tensorboard**: 本地可视化
- **自动管理**: 实验目录和run name自动有序化

---

## 常见问题

### 环境配置
**Q: 如何切换日志系统？**
A: 修改config中的`LOGGER`字段即可，无需改动主代码。

**Q: CUDA内存不足怎么办？**
A: 减少batch_size，或使用更小的模型。

### 训练相关
**Q: 如何自定义多阶段流程？**
A: 编辑`multistage_train_config.py`或`multistage_finetune_config.py`，按注释填写每阶段参数。

**Q: 如何复现某次实验？**
A: 日志目录和wandb run name均带有时间戳和任务名，便于查找和复现。

### 主动学习
**Q: 主动学习内存不足怎么办？**
A: 减少`batch_size`和`BATCH_SIZE_TO_LABEL`，或使用更小的模型。

**Q: 主动学习选择效果不佳？**
A: 调整不确定性策略权重，启用多样性策略，或检查特征提取方法。

### 模型评估
**Q: 如何评估模型性能？**
A: 使用`python tasks/evaluation_task.py`进行统一评估，支持可视化和定量分析。

---

## 贡献指南

欢迎提交Issue或Pull Request来改进项目！

### 开发环境设置
1. Fork项目
2. 创建功能分支: `git checkout -b feature/AmazingFeature`
3. 提交更改: `git commit -m 'Add some AmazingFeature'`
4. 推送分支: `git push origin feature/AmazingFeature`
5. 提交Pull Request

### 代码规范
- 遵循PEP 8代码风格
- 添加适当的注释和文档
- 确保所有测试通过

---

## 联系我们

如果您有特殊硬件/平台需求、想集成新模型或有大规模实验需求，欢迎联系开发团队。

---

## 致谢

感谢所有为这个项目做出贡献的开发者和研究人员！

---

*最后更新时间: 2025年8月*


