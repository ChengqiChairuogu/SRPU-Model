# SRPU-Model 项目结构说明

## 项目概述
SRPU-Model 是一个面向扫描电镜（SEM）图像分割的自进化深度学习平台，集成了监督学习、自监督学习、半监督学习和主动学习等多种策略。

## 完整目录结构

```
SRPU-Model/
├── README.md                                    # 项目说明文档
├── STRUCTURE.md                                 # 结构说明文档
├── environment.yml                              # Conda环境依赖配置
├── .git/                                       # Git版本控制
├── .ipynb_checkpoints/                        # Jupyter临时文件
│
├── configs/                                    # 配置文件目录
│   ├── __init__.py
│   ├── base.py                                # 基础配置
│   ├── augmentation_config.py                 # 数据增强配置
│   ├── dataset_config.py                      # 数据集配置
│   ├── json_config.py                         # JSON生成配置
│   ├── wandb_config.py                        # WandB日志配置
│   ├── __pycache__/
│   ├── .ipynb_checkpoints/
│   │
│   ├── active_learning/                       # 主动学习配置
│   │   ├── active_learning_config.py
│   │   └── sam_config.py
│   │
│   ├── train/                                 # 训练配置
│   │   ├── __init__.py
│   │   ├── train_config.py
│   │   ├── multistage_train_config.py
│   │   ├── __pycache__/
│   │   └── .ipynb_checkpoints/
│   │
│   ├── finetune/                              # 微调配置
│   │   ├── __init__.py
│   │   ├── finetune_config.py
│   │   ├── multistage_finetune_config.py
│   │   ├── __pycache__/
│   │   └── .ipynb_checkpoints/
│   │
│   ├── inference/                             # 推理配置
│   │   ├── __init__.py
│   │   ├── inference_config.py
│   │   ├── evaluation_config.py
│   │   ├── __pycache__/
│   │   └── .ipynb_checkpoints/
│   │
│   ├── selfup/                                # 自监督学习配置
│   │   ├── __init__.py
│   │   ├── ssl_config.py
│   │   ├── ssl_inspect_config.py
│   │   ├── __pycache__/
│   │   └── .ipynb_checkpoints/
│   │
│   └── image_sharpness/                       # 图像清晰度配置
│       ├── image_sharpness_config.py
│       ├── multi_sharpness_training_config.py
│       └── sharpness_processor_config.py
│
├── tasks/                                     # 任务脚本目录
│   ├── __init__.py
│   ├── active_learning_task.py               # 主动学习任务
│   ├── enhanced_active_learning_task.py      # 增强版主动学习任务
│   ├── ssl_pretrain_task.py                  # 自监督预训练任务
│   ├── ssl_inspect_task.py                   # 自监督模型检查任务
│   ├── train_task.py                         # 监督训练任务
│   ├── finetune_task.py                      # 模型微调任务
│   ├── evaluation_task.py                    # 模型评估任务
│   ├── inference_task.py                     # 模型推理任务
│   └── image_sharpness_ssl_task.py          # 图像清晰度任务
│
├── pipelines/                                 # 多阶段训练管道目录
│   ├── multistage_train_pipeline.py          # 多阶段训练管道
│   ├── multistage_finetune_pipeline.py       # 多阶段微调管道
│   └── multi_sharpness_training_pipeline.py  # 多清晰度训练管道
│
├── utils/                                     # 工具函数目录
│   ├── __init__.py
│   ├── augmentation_util.py                   # 数据增强工具
│   ├── dataset_statistics_calculation_util.py # 数据集统计计算
│   ├── stratified_sample_util.py             # 分层采样工具
│   ├── uncertainty_util.py                   # 不确定性计算工具
│   ├── training_util.py                      # 训练工具函数
│   ├── logging_util.py                       # 日志工具
│   ├── image_sizes_util.py                   # 图像尺寸检查工具
│   ├── image_sharpness_util.py               # 图像清晰度工具
│   ├── sam_annotation_util.py                # SAM标注工具
│   ├── smart_annotation_util.py              # 智能标注工具
│   ├── pretrained_model_util.py              # 预训练模型工具
│   ├── analysis_chart_util.py                # 分析图表工具
│   ├── __pycache__/
│   └── .ipynb_checkpoints/
│
├── tools/                                     # 工具脚本目录
│   ├── __init__.py
│   ├── json_generator.py                     # JSON索引生成器
│   ├── image_sharpness_processor.py          # 图像清晰度处理器
│   ├── pretrained_model_manager.py           # 预训练模型管理器
│   ├── __pycache__/
│   └── .ipynb_checkpoints/
│
├── data/                                     # 数据目录
│   ├── raw/                                 # 原始图像注册数据
│   ├── labeled/                             # 已标注数据
│   ├── inference/                           # 推理数据
│   │   ├── input/                          # 推理输入
│   │   └── output/                         # 推理输出
│   └── .ipynb_checkpoints/
│
├── datasets/                                 # 数据集定义目录
│   ├── __init__.py
│   ├── sem_datasets.py                      # SEM分割数据集
│   ├── ssl_dataset.py                       # 自监督数据集
│   ├── image_sharpness_dataset.py           # 图像清晰度数据集
│   ├── __pycache__/
│   ├── .ipynb_checkpoints/
│   │
│   ├── dataset1_LInCl/                     # 数据集1
│   │   ├── raw_images/                     # 原始图像
│   │   └── masks_3class/                   # 3类掩码标注
│   │
│   ├── dataset2_LPSCl/                     # 数据集2
│   │   ├── raw_images/
│   │   └── masks_3class/
│   │
│   └── dataset3_LNOCl/                     # 数据集3
│       ├── raw_images/
│       └── masks_3class/
│
├── json/                                    # JSON索引文件目录
│   ├── master_labeled_dataset.json          # 主标注数据集索引
│   ├── master_ssl_dataset.json              # 主自监督数据集索引
│   ├── dataset1_LInCl.json                 # 数据集1索引
│   ├── dataset1_LInCl_ssl.json             # 数据集1自监督索引
│   ├── dataset2_LPSCl.json                 # 数据集2索引
│   ├── dataset2_LPSCl_ssl.json             # 数据集2自监督索引
│   ├── dataset3_LNOCl.json                 # 数据集3索引
│   └── dataset3_LNOCl_ssl.json             # 数据集3自监督索引
│
├── models/                                  # 模型定义目录
│   ├── __init__.py
│   ├── segmentation_unet.py                 # 分割UNet模型
│   ├── ssl_unet_autoencoder.py             # UNet自编码器（自监督学习）
│   ├── ssl_mae_model.py                    # MAE自监督模型
│   ├── image_sharpness_unet.py             # 图像清晰度UNet模型
│   ├── __pycache__/
│   ├── .ipynb_checkpoints/
│   ├── checkpoints/                         # 模型检查点
│   │
│   ├── encoders/                           # 编码器模块
│   │   ├── __init__.py
│   │   ├── efficientnet_encoder.py         # EfficientNet编码器
│   │   ├── dinov2_encoder.py               # DINOv2编码器
│   │   ├── __pycache__/
│   │   └── .ipynb_checkpoints/
│   │
│   └── decoders/                           # 解码器模块
│       ├── __init__.py
│       ├── unet_decoder.py                 # UNet解码器
│       ├── deeplab_decoder.py              # DeepLab解码器
│       ├── __pycache__/
│       └── .ipynb_checkpoints/
│
├── active_learning/                         # 主动学习输出目录
│   ├── iteration_1/                         # 第1轮迭代
│   │   ├── LPSCl-051.png                   # 选择的样本图像
│   │   └── LPSCl-051_prediction_mask.png   # 预测mask
│   └── ...                                 # 更多迭代目录
│
└── docs/                                    # 项目文档目录
    ├── ENHANCED_ACTIVE_LEARNING_GUIDE.md   # 增强主动学习指南
    ├── MULTI_SHARPNESS_OUTPUT_STRUCTURE_UPDATE.md  # 多清晰度输出结构更新
    └── SAM_SETUP_GUIDE.md                  # SAM设置指南
```

## 主要模块说明

### 配置模块 (configs/)
- **基础配置**: `base.py` - 全局参数设置
- **主动学习**: `active_learning/` - 主动学习相关配置，包括SAM配置
- **训练配置**: `train/` - 监督训练和多阶段训练配置
- **微调配置**: `finetune/` - 模型微调配置
- **推理配置**: `inference/` - 推理和检查配置
- **自监督配置**: `selfup/` - 自监督学习配置
- **图像清晰度配置**: `image_sharpness/` - 图像清晰度相关配置

### 任务模块 (tasks/)
- **主动学习任务**:
  - `active_learning_task.py` - 主动学习迭代任务
  - `enhanced_active_learning_task.py` - 增强版主动学习任务（集成SAM）
- **自监督学习任务**:
  - `ssl_pretrain_task.py` - 自监督预训练任务
  - `ssl_inspect_task.py` - 自监督模型检查
- **核心任务**:
  - `train_task.py` - 基础监督训练
  - `finetune_task.py` - 模型微调任务
  - `evaluation_task.py` - 模型性能评估
  - `inference_task.py` - 模型推理任务
- **图像清晰度任务**:
  - `image_sharpness_ssl_task.py` - 图像清晰度评估任务

### 管道模块 (pipelines/)
- **多阶段训练**: `multistage_train_pipeline.py` - 多阶段训练管道
- **多阶段微调**: `multistage_finetune_pipeline.py` - 多阶段微调管道
- **多清晰度训练**: `multi_sharpness_training_pipeline.py` - 多清晰度训练管道

### 工具模块 (utils/)
- **数据增强**: `augmentation_util.py` - 数据增强工具
- **统计计算**: `dataset_statistics_calculation_util.py` - 数据集统计
- **分层采样**: `stratified_sample_util.py` - 分层采样
- **不确定性**: `uncertainty_util.py` - 不确定性计算
- **训练工具**: `training_util.py` - 训练相关函数
- **日志工具**: `logging_util.py` - 日志记录
- **图像检查**: `image_sizes_util.py` - 图像尺寸检查
- **图像清晰度**: `image_sharpness_util.py` - 图像清晰度工具
- **SAM标注**: `sam_annotation_util.py` - SAM标注工具
- **智能标注**: `smart_annotation_util.py` - 智能标注工具
- **预训练模型**: `pretrained_model_util.py` - 预训练模型工具
- **分析图表**: `analysis_chart_util.py` - 分析图表工具

### 工具脚本模块 (tools/)
- **JSON生成**: `json_generator.py` - JSON索引生成器
- **图像清晰度处理**: `image_sharpness_processor.py` - 图像清晰度处理器
- **预训练模型管理**: `pretrained_model_manager.py` - 预训练模型管理器

### 数据模块
- **原始数据**: `data/raw/` - 无标签原始图像
- **标注数据**: `data/labeled/` - 已标注数据
- **推理数据**: `data/inference/` - 推理输入输出
- **数据集定义**: `datasets/` - PyTorch数据集类
- **JSON索引**: `json/` - 数据集索引文件
- **主动学习输出**: `active_learning/` - 主动学习预测和选择信息

### 模型模块 (models/)
- **分割模型**: `segmentation_unet.py` - 主要分割模型
- **自编码器**: `ssl_unet_autoencoder.py` - UNet自编码器（自监督学习）
- **MAE模型**: `ssl_mae_model.py` - Masked Autoencoder（自监督学习）
- **图像清晰度模型**: `image_sharpness_unet.py` - 图像清晰度UNet模型
- **编码器**: `encoders/` - 各种编码器实现
- **解码器**: `decoders/` - 各种解码器实现
- **检查点**: `checkpoints/` - 模型权重保存

### 文档模块 (docs/)
- **增强主动学习指南**: `ENHANCED_ACTIVE_LEARNING_GUIDE.md`
- **多清晰度输出结构更新**: `MULTI_SHARPNESS_OUTPUT_STRUCTURE_UPDATE.md`
- **SAM设置指南**: `SAM_SETUP_GUIDE.md`

## 核心功能

1. **主动学习**: 通过不确定性采样选择最有价值的样本进行标注
   - 支持多种不确定性计算方法（熵、置信度、边缘等）
   - 自动生成预测mask用于手动标注调整
   - 按迭代轮数组织输出文件（`active_learning/iteration_*/`）
2. **自监督学习**: 使用MAE等方法进行预训练
3. **监督学习**: 基于标注数据进行分割训练
4. **多阶段训练**: 支持复杂的多阶段训练流程
5. **图像清晰度评估**: 自动评估图像清晰度质量
6. **多清晰度训练**: 支持不同清晰度级别的模型训练
7. **SAM集成**: 支持Segment Anything Model标注
8. **模型评估**: 全面的模型性能评估
9. **推理部署**: 支持模型推理和结果分析

## 使用说明

1. **环境配置**: 使用 `environment.yml` 创建conda环境
2. **数据准备**: 将数据放入相应目录，运行 `tools/json_generator.py` 生成索引
3. **模型训练**: 选择合适的任务脚本进行训练
4. **主动学习**: 运行 `tasks/active_learning_task.py` 开始主动学习流程
5. **图像清晰度评估**: 运行 `tasks/image_sharpness_ssl_task.py` 评估图像清晰度
6. **多清晰度训练**: 运行 `pipelines/multi_sharpness_training_pipeline.py` 进行多清晰度训练
7. **模型评估**: 使用 `tasks/evaluation_task.py` 评估模型性能
8. **推理部署**: 使用 `tasks/inference_task.py` 进行推理

## 注意事项

- 所有配置文件都放在 `configs/` 目录下
- 工具函数统一使用 `_util` 后缀命名
- 工具脚本放在 `tools/` 目录下
- 支持多种编码器和解码器组合
- 提供完整的主动学习流程
- 支持多数据集训练和评估
- 集成图像清晰度评估和多清晰度训练功能
- 支持SAM标注工具