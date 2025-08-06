# SRPU-Model 项目说明

## 项目简介

SRPU-Model 是一个面向扫描电镜（SEM）图像分割的自进化深度学习平台，集成了**监督学习**、**自监督学习**、**半监督学习**和**主动学习**等多种策略，目标是用有限标注数据和大量无标注数据，持续提升分割模型性能，实现"自我成长"的智能分割系统。

---

## 目录结构与详细说明

```
SRPU-Model/
├── configs/         # 配置文件（训练、微调、自监督、推理、多阶段等）
├── data/            # 数据集（raw原图、labeled标注、inference推理等）
├── datasets/        # PyTorch数据集定义及原始/掩码图片
├── json/            # 数据集索引json文件
├── models/          # 模型结构与权重
├── tasks/           # 各类训练/推理/评估脚本
├── pipelines/       # 多阶段训练/微调主入口脚本
├── utils/           # 工具函数与脚本
├── tensorboard/     # tensorboard日志目录
├── wandb/           # wandb日志目录（如存在）
├── environment.yml  # Conda环境依赖
├── README.md        # 项目说明
└── STRUCTURE.md     # 结构说明
```

### configs/ 配置文件目录
- **base.py**：全局基础配置，如路径、类别数、图像尺寸等。
- **augmentation_config.py**：数据增强参数。
- **dataset_config.py**：数据集划分比例、随机种子。
- **json_config.py**：json索引生成相关配置。
- **wandb_config.py**：wandb日志系统全局配置。
- **finetune/**
  - `finetune_config.py`：单阶段微调参数。
  - `multistage_finetune_config.py`：多阶段微调参数。
- **train/**
  - `train_config.py`：单阶段有监督训练参数。
  - `multistage_train_config.py`：多阶段有监督训练参数。
- **selfup/**
  - `ssl_config.py`：自监督预训练参数。
  - `ssl_inspect_config.py`：自监督模型评估参数。
- **inference/**
  - `inference_config.py`：推理参数。
  - `evaluation_config.py`：模型评估和推理结果分析参数。
- **active_learning/**
  - `active_learning_config.py`：主动学习核心参数配置。

### data/ 数据目录
- **raw/**：原始未标注SEM图像，按实验/批次分类存放。
- **labeled/**：已标注的图像及其掩码，文件名与raw一致。
- **inference/**
  - `input/`：待推理图片。
  - `output/`：推理生成的分割掩码。

### datasets/ 数据集定义
- **dataset1_LInCl/**、**dataset2_LPSCl/**、**dataset3_LNOCl/**
  - `raw_images/`：原始图片，按数据集分组。
  - `masks_3class/`：三分类掩码，按数据集分组。
- **sem_datasets.py**：主分割数据集类，支持多数据集联合、分层采样。
- **ssl_dataset.py**：自监督/半监督任务专用数据集类。

### json/ 数据集索引
- 各种json文件（如`master_labeled_dataset.json`、`dataset1_LInCl.json`等）：记录所有样本的路径、标签、所属数据集等信息，供训练/推理脚本快速索引。

### models/ 模型结构与权重
- **encoders/**
  - `efficientnet_encoder.py`、`dinov2_encoder.py`等：不同主流编码器实现。
- **decoders/**
  - `unet_decoder.py`、`deeplab_decoder.py`等：不同主流解码器实现。
- **segmentation_unet.py**：U-Net分割模型主结构。
- **unet_autoencoder.py**：U-Net自编码器结构（自监督用）。
- **mae_model.py**：MAE自监督模型结构。
- **checkpoints/**
  - `ssl_pretrained_unet/`等：各类模型的权重保存目录。

### tasks/ 任务脚本
- **train_task.py**：单阶段有监督训练主入口。
- **finetune_task.py**：单阶段微调主入口。
- **ssl_pretrain_task.py**：自监督预训练主入口。
- **ssl_inspect_task.py**：自监督模型评估脚本。
- **inference_task.py**：推理脚本，将input图片分割并输出掩码。
- **evaluation_task.py**：统一模型评估脚本（合并了原有的inspect_validation_results.py和difference_task.py）。
- **active_learning_task.py**：主动学习主入口脚本。

### pipelines/ 多阶段训练/微调主入口
- **multistage_train_pipeline.py**：多阶段有监督训练主入口。
- **multistage_finetune_pipeline.py**：多阶段微调主入口。

### utils/ 工具函数
- **augmentation_util.py**：数据增强实现。
- **dataset_statistics_calculation_util.py**：统计数据集分布、类别比例。
- **image_sizes_util.py**：检查图片尺寸一致性。
- **json_generator_util.py**：自动生成/更新json索引文件。
- **logging_util.py**：日志系统适配（wandb/tensorboard），自动管理日志目录和run name。
- **stratified_sample_util.py**：分层采样器实现，保证各子数据集均衡采样。
- **training_util.py**：训练通用工具（损失函数、评估、训练循环等）。
- **uncertainty_util.py**：不确定性计算和多样性选择工具。

### 其它重要文件
- **environment.yml**：Conda环境依赖说明。
- **README.md**：项目说明文档。
- **STRUCTURE.md**：代码结构和扩展说明。

---

## 环境配置

推荐使用 Conda 环境：

```bash
conda env create -f environment.yml
conda activate SRPU-Model
```

---

## 数据准备

1. 原始SEM图像放入 `data/raw/`
2. 标注掩码放入 `data/labeled/`，文件名需与原图一致（仅扩展名不同）

---

## 数据集索引生成

自动生成数据集json索引：

```bash
python utils/json_generator_util.py --mode generate_all
```

---

## 日志系统与实验管理

### 日志系统选择
- 在各类 config 文件（如 `configs/train/train_config.py`、`configs/finetune/finetune_config.py`、`configs/selfup/ssl_config.py`、多阶段config等）中通过 `LOGGER` 字段选择日志方式：

  ```python
  LOGGER = "wandb"  # 可选 "wandb" 或 "tensorboard"
  ```

- **日志目录结构已标准化为：**
  - `tensorboard/`  （如 tensorboard 日志）
  - `wandb/`        （如 wandb 日志）
  - 每次实验会自动在 tensorboard 或 wandb 下新建唯一子目录，便于管理和查找。

### 日志查看
- **Wandb**：支持在线/离线模式，详见 [wandb 官网](https://wandb.ai/)。
- **Tensorboard**：如选择 tensorboard，运行：

  ```bash
  tensorboard --logdir tensorboard
  ```

  浏览 http://localhost:6006 查看训练曲线。

---

## 训练与微调

### 单阶段训练/微调/自监督
- **监督训练**  
  ```bash
  python tasks/train_task.py
  ```
- **微调**  
  ```bash
  python tasks/finetune_task.py
  ```
- **自监督预训练**  
  ```bash
  python tasks/ssl_pretrain_task.py
  ```

### 模型评估

统一评估脚本，支持定性和定量评估：

```bash
python tasks/evaluation_task.py
```

该脚本合并了原有的可视化评估和指标计算功能，提供：
- 预测结果可视化（原图、真值、预测对比）
- 定量指标计算（Dice Score、IoU Score）
- 结果保存和统计

### 主动学习

主动学习模块旨在通过智能选择最有价值的样本进行人工标注，最大化标注效率，减少标注成本。

#### 核心特性

1. **多种不确定性采样策略**
   - **熵不确定性 (Entropy Uncertainty)**: 基于预测概率分布的熵
   - **置信度不确定性 (Confidence Uncertainty)**: 基于预测的最大置信度
   - **边缘不确定性 (Margin Uncertainty)**: 基于最高概率与次高概率的差值
   - **集成不确定性 (Ensemble Uncertainty)**: 基于多个模型的预测方差

2. **多样性采样策略**
   - **核心集选择 (Core Set Selection)**: 确保选择的样本具有代表性
   - **聚类选择 (Cluster-based Selection)**: 基于聚类的多样性采样

3. **智能数据管理**
   - 自动备份原始数据
   - 动态管理已标注和未标注数据
   - 详细的选择历史记录

4. **预测Mask生成** ⭐ **新功能**
   - 自动为最不确定样本生成预测mask
   - 支持彩色mask输出，便于可视化
   - 可直接用于手动标注调整

5. **可视化支持**
   - 不确定性热力图生成
   - 选择样本的可视化
   - 训练过程监控

#### 使用方法

1. **配置参数**

编辑 `configs/active_learning/active_learning_config.py` 文件：

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

2. **运行主动学习**

```bash
python tasks/active_learning_task.py
```

3. **工作流程**

   1. **初始化**: 系统自动备份原始数据，创建初始训练集
   2. **训练**: 使用当前已标注数据训练模型
   3. **不确定性计算**: 对未标注样本计算不确定性分数
   4. **样本选择**: 基于不确定性和多样性策略选择样本
   5. **人工标注**: 用户对选中的样本进行标注
   6. **数据更新**: 将新标注的样本添加到训练集
   7. **重复**: 重复步骤2-6直到满足停止条件

#### 输出结果

1. **检查点文件**
```
models/checkpoints/active_learning_unet_unet/
├── round_1_best.pth          # 第1轮最佳模型
├── round_1_final.pth         # 第1轮最终模型
├── round_2_best.pth          # 第2轮最佳模型
├── ...
├── training_history.json      # 训练历史
└── final_statistics.json     # 最终统计
```

2. **可视化结果**
```
active_learning_results/
├── round_1/
│   ├── sample1_uncertainty.png    # 不确定性热力图
│   ├── sample2_uncertainty.png
│   └── selection_summary.json     # 选择摘要
├── round_2/
│   └── ...
```

3. **预测Mask输出** ⭐ **新功能**
```
active_learning_predictions/
├── round_1/
│   ├── sample1_predicted_mask.png    # 预测mask
│   ├── sample2_predicted_mask.png
│   └── selection_info.json           # 选择信息
├── round_2/
│   └── ...
```

4. **数据管理**
```
json/
├── backups/                    # 原始数据备份
├── round_1/                   # 第1轮数据
├── round_2/                   # 第2轮数据
└── active_learning_history.json  # 选择历史
```

#### 预测Mask生成功能 ⭐ **新功能**

主动学习系统现在支持自动生成预测mask，方便您进行手动标注调整。

**使用方法：**

1. **运行主动学习（自动生成预测mask）**
```bash
python tasks/active_learning_task.py
```

2. **单独生成预测mask**
```bash
python tasks/generate_prediction_masks.py
```

3. **手动标注预测mask**
```bash
# 下载预测文件到本地
# 使用ImageJ等工具修改预测mask
# 上传修改后的mask到 data/labeled/
```

**工作流程：**
1. 系统自动选择最不确定的样本
2. 生成预测mask并保存为PNG格式
3. 使用ImageJ等工具手动调整边界
4. 将修改后的标注加入数据集

详细使用指南请参考：`ACTIVE_LEARNING_MASK_GUIDE.md`

#### 配置说明

**预测Mask配置**
```python
PREDICTION_MASK_CONFIG = {
    "generate_prediction_masks": True,  # 是否生成预测mask
    "save_colored_masks": True,        # 保存彩色mask
    "output_dir": "active_learning_predictions"  # 输出目录
}

# 颜色映射和类别名称统一使用base.py中的配置
# COLOR_MAPPING = {0: [0, 0, 255], 1: [0, 255, 255], 2: [255, 126, 126]}
# CLASS_NAMES = {0: "carbon", 1: "SE", 2: "AM"}
```

**不确定性策略**
```python
UNCERTAINTY_STRATEGIES = {
    "entropy": {
        "enabled": True,        # 是否启用
        "weight": 1.0,          # 权重
        "description": "基于预测概率分布的熵"
    },
    "confidence": {
        "enabled": True,
        "weight": 1.0,
        "description": "基于预测的最大置信度"
    },
    "margin": {
        "enabled": True,
        "weight": 1.0,
        "description": "基于最高概率与次高概率的差值"
    }
}
```

**多样性策略**
```python
DIVERSITY_STRATEGIES = {
    "core_set": {
        "enabled": True,        # 是否启用
        "weight": 0.5,          # 权重
        "description": "基于核心集选择的多样性"
    },
    "cluster_based": {
        "enabled": True,
        "weight": 0.3,
        "description": "基于聚类的多样性"
    }
}
```

**训练参数**
```python
TRAINING_CONFIG = {
    "num_epochs_per_round": 100,  # 每轮训练的epoch数
    "batch_size": 8,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "optimizer": "AdamW",
    "loss_function": "DiceCELoss"
}
```

#### 停止条件

主动学习会在以下情况下停止：

1. **达到最大轮数**: `MAX_ACTIVE_LEARNING_ROUNDS`
2. **性能提升不足**: 连续轮次的性能提升低于 `MIN_IMPROVEMENT_THRESHOLD`
3. **无未标注样本**: 所有样本都已标注

#### 监控和调试

1. **训练监控**
   - 每轮训练都会显示训练损失和验证Dice系数
   - 支持早停机制，避免过拟合

2. **不确定性分析**
   - 生成不确定性热力图，直观显示模型的不确定区域
   - 记录不确定性分数分布

3. **选择历史**
   - 详细记录每轮选择的样本信息
   - 包含不确定性分数和选择排名

#### 扩展功能

1. **自定义不确定性策略**
在 `utils/uncertainty_util.py` 中添加新的不确定性计算方法：

```python
def calculate_custom_uncertainty(self, logits: torch.Tensor) -> torch.Tensor:
    # 实现自定义不确定性计算
    pass
```

2. **自定义多样性策略**
在 `utils/uncertainty_util.py` 中添加新的多样性选择方法：

```python
def custom_diversity_selection(self, features: np.ndarray, batch_size: int) -> List[int]:
    # 实现自定义多样性选择
    pass
```

3. **集成学习**
启用集成不确定性策略，使用多个模型进行预测：

```python
UNCERTAINTY_STRATEGIES = {
    "ensemble": {
        "enabled": True,
        "weight": 1.0,
        "description": "基于集成模型的预测方差"
    }
}
```

#### 注意事项

1. **数据备份**: 系统会自动备份原始数据，确保数据安全
2. **内存使用**: 大量样本的不确定性计算可能消耗较多内存
3. **计算时间**: 特征提取和不确定性计算可能需要较长时间
4. **标注质量**: 人工标注的质量直接影响模型性能

#### 故障排除

1. **内存不足**
   - 减少 `batch_size`
   - 减少 `BATCH_SIZE_TO_LABEL`
   - 使用更小的模型

2. **训练不收敛**
   - 调整学习率
   - 增加训练轮数
   - 检查数据质量

3. **选择效果不佳**
   - 调整不确定性策略权重
   - 启用多样性策略
   - 检查特征提取方法

#### 与项目其他模块的集成

主动学习模块与项目的其他模块紧密集成：

1. **自监督学习**: 可以使用预训练的编码器初始化模型
2. **半监督学习**: 可以与伪标签策略结合
3. **数据增强**: 使用项目统一的数据增强策略
4. **评估系统**: 使用项目统一的评估指标

通过这种集成，主动学习模块能够充分利用项目已有的基础设施，实现高效的样本选择和数据管理。

### 多阶段训练/微调
- **多阶段有监督训练**  
  配置：`configs/train/multistage_train_config.py`  
  运行：  
  ```bash
  python pipelines/multistage_train_pipeline.py
  ```
- **多阶段微调**  
  配置：`configs/finetune/multistage_finetune_config.py`  
  运行：  
  ```bash
  python pipelines/multistage_finetune_pipeline.py
  ```
- 多阶段流程支持每阶段自定义数据集、epoch、学习率、权重加载等，详见对应config文件注释。

---

## 推理

将待分割图片放入 `data/inference/input/`，运行：

```bash
python tasks/inference_task.py
```

分割结果会保存在 `data/inference/output/`。

---

## 检查点与模型保存

- 所有模型权重和断点都保存在 `models/checkpoints/` 下，最优模型为 `best_model.pth`。
- 多阶段任务每阶段会自动保存独立的最佳模型。
- 你可以通过 config 灵活指定保存路径。

---

## 高级特性

- 支持多种编码器（U-Net、ResNet、EfficientNet、DINOv2等）和解码器（U-Net、DeepLabV3+）
- 支持断点续训、自动保存最优模型
- 支持自监督预训练、半监督伪标签、主动学习等扩展
- 支持多阶段训练/微调，灵活组合数据集与训练策略
- 支持主动学习，智能选择最有价值的样本进行人工标注
- 支持多种不确定性采样策略（熵、置信度、边缘、集成）
- 支持多样性采样策略（核心集、聚类）
- 支持统一模型评估（定性和定量分析）
- 日志与实验目录自动有序化，便于大规模实验管理

---

## 贡献与扩展

欢迎提交 issue 或 PR，或根据 `STRUCTURE.md` 了解如何扩展新模型、数据集或任务。

---

## 常见问题

- **如何切换日志系统？**  
  修改config中的`LOGGER`字段即可，无需改动主代码。
- **如何自定义多阶段流程？**  
  编辑`multistage_train_config.py`或`multistage_finetune_config.py`，按注释填写每阶段参数即可。
- **如何复现某次实验？**  
  日志目录和wandb run name均带有时间戳和任务名，便于查找和复现。
- **如何配置主动学习？**  
  编辑`configs/active_learning/active_learning_config.py`，调整不确定性策略、多样性策略和训练参数。
- **主动学习内存不足怎么办？**  
  减少`batch_size`和`BATCH_SIZE_TO_LABEL`，或使用更小的模型。
- **主动学习选择效果不佳？**  
  调整不确定性策略权重，启用多样性策略，或检查特征提取方法。
- **如何评估模型性能？**  
  使用`python tasks/evaluation_task.py`进行统一评估，支持可视化和定量分析。

---

如需详细使用说明、参数解释或遇到问题，欢迎随时提问！

---

如果你有特殊硬件/平台需求、想集成新模型或有大规模实验需求，也欢迎联系开发者团队。

### 主动学习模块
- **功能**：实现不确定性采样和模型迭代更新。
- **主要文件**：tasks/active_learning_task.py, configs/active_learning/active_learning_config.py。
- **运行**：python tasks/active_learning_task.py

---

## 详细使用指南

### 主动学习使用指南

#### 概述
主动学习模块旨在通过智能选择最有价值的样本进行人工标注，最大化标注效率，减少标注成本。本指南介绍如何使用修改后的主动学习策略，每次只标注一张图像，实现精细化的模型优化。

#### 配置参数

**主动学习参数**
- `NUM_ITERATIONS = 20`: 迭代轮数（可根据需要调整）
- `SAMPLES_PER_ITER = 1`: 每轮选择的样本数（固定为1）
- `UNCERTAINTY_METHODS = ["entropy"]`: 不确定性计算方法

**训练参数**
- `BATCH_SIZE = 8`: 批次大小
- `LEARNING_RATE = 1e-4`: 学习率
- `NUM_EPOCHS_PER_ITER = 10`: 每轮微调的epoch数

#### 使用流程

**1. 准备数据**
确保以下目录结构正确：
```
data/
├── raw/          # 无标签图像池
└── labeled/      # 已标注图像
```

**2. 运行主动学习**
```bash
python -m tasks.active_learning_task
```

**3. 标注流程**
每轮迭代时：
1. 系统会显示选中的图像信息
2. 显示不确定性分数和图像路径
3. 系统自动生成预测mask图像
4. 用户使用ImageJ等工具手动调整mask
5. 将标注完成的图像移动到 `data/labeled/` 目录
6. 按 Enter 继续下一轮

**4. 监控指标**
系统会显示：
- 当前标注进度
- 模型性能改进
- 不确定性分数范围
- 性能历史趋势

#### 优势

**精细化控制**
- 每次只标注一张图像，可以精确控制标注质量
- 能够观察每张新图像对模型性能的影响

**智能预测辅助**
- 自动生成模型预测结果作为标注起点
- 支持手动标注调整，使用ImageJ等工具
- 减少从头开始标注的工作量

**实时反馈**
- 显示不确定性分数，了解模型对图像的置信度
- 实时显示性能改进情况

**灵活配置**
- 可以根据需要调整迭代轮数
- 支持多种不确定性计算方法

#### 注意事项

1. **标注质量**: 确保每张图像的标注质量，因为单张图像的影响更大
2. **数据平衡**: 注意标注数据的类别平衡
3. **性能监控**: 关注性能改进趋势，如果连续多轮没有改进，可能需要调整策略
4. **图像编辑工具**: 确保安装了ImageJ等图像编辑工具用于标注调整

#### 环境准备

**安装ImageJ**
- 下载地址：https://imagej.nih.gov/ij/
- 或使用其他图像编辑软件：GIMP、Photoshop、Paint.NET等

#### 输出文件

- `active_learning/selection_info/`: 每轮选择的样本信息
- `active_learning/checkpoints/`: 模型检查点
- `active_learning/predictions/iteration_*/`: 每轮迭代的预测mask和原始图像
- `runs/tensorboard/`: 训练日志

#### 自定义配置

可以根据需要修改 `configs/active_learning/active_learning_config.py` 中的参数：

```python
# 增加迭代轮数
NUM_ITERATIONS = 30

# 修改不确定性方法
UNCERTAINTY_METHODS = ["entropy", "margin"]

# 调整训练参数
NUM_EPOCHS_PER_ITER = 15
```

### 手动标注指南

#### 概述
本指南介绍如何使用ImageJ等工具对主动学习生成的预测结果进行手动标注。

#### 工作流程

**1. 在服务器上运行主动学习**
```bash
python -m tasks.active_learning_task
```

**2. 查看预测文件**
```bash
# 查看所有迭代目录
ls active_learning/predictions/

# 查看特定迭代的文件
ls active_learning/predictions/iteration_1/
```

**3. 下载预测文件到本地**
```bash
# 从服务器下载特定迭代的预测文件
scp -r username@server_ip:~/SRPU-Model/active_learning/predictions/iteration_1/ ./

# 或下载所有迭代的文件
scp -r username@server_ip:~/SRPU-Model/active_learning/predictions/ ./
```

**4. 使用ImageJ进行标注**
```bash
# 打开ImageJ
# 加载预测mask图像
# 使用画笔工具修正边界
# 应用形态学操作平滑边缘
# 保存修改后的mask
```

**5. 上传标注文件**
```bash
# 将修改后的mask上传回服务器
scp image_name_mask.png username@server_ip:~/SRPU-Model/data/labeled/

# 将原始图像也移动到标注目录
scp image_name.png username@server_ip:~/SRPU-Model/data/labeled/
```

#### 详细步骤

**步骤1: 服务器端准备**
1. 运行主动学习任务
2. 系统自动生成预测mask图像
3. 查看预测文件列表

**步骤2: 本地环境准备**
1. 安装ImageJ或其他图像编辑软件
2. 准备足够的磁盘空间存储预测文件
3. 确保网络连接稳定

**步骤3: 下载和准备**
1. 使用scp下载预测文件
2. 检查文件完整性（原始图像和预测mask）
3. 准备标注工具

**步骤4: 标注工作**
1. 打开ImageJ应用程序
2. 加载预测mask图像
3. 查看模型预测结果
4. 使用画笔工具修正边界
5. 应用形态学操作平滑边缘
6. 保存修改后的mask

**步骤5: 上传和同步**
1. 将标注完成的mask文件上传回服务器
2. 在服务器上将文件移动到标注目录
3. 继续下一轮主动学习迭代

#### 标注技巧

**高效标注**
- **利用预测结果**: 模型预测通常已经相当准确，只需要微调
- **批量处理**: 可以同时处理多个预测文件
- **快捷键**: 熟练使用ImageJ的快捷键提高效率

**质量控制**
- **边界精度**: 确保边界与图像特征精确对齐
- **类别一致性**: 保持类别标签的一致性
- **平滑处理**: 使用形态学操作平滑边界

**文件管理**
- **备份**: 定期备份重要的标注结果
- **版本控制**: 使用版本控制管理标注文件
- **命名规范**: 保持文件命名的一致性

#### 故障排除

**常见问题**

**1. 无法下载文件**
```bash
# 检查网络连接
ping server_ip

# 检查文件是否存在
ssh username@server_ip "ls -la ~/SRPU-Model/active_learning/predictions/iteration_1/"
```

**2. ImageJ无法启动**
- 检查ImageJ安装是否正确
- 确保Java环境配置正确
- 尝试重新安装ImageJ

**3. 文件损坏**
```bash
# 检查文件完整性
md5sum *.png

# 重新下载文件
scp -r username@server_ip:~/SRPU-Model/active_learning/predictions/iteration_1/ ./
```

**4. 标注丢失**
- 定期保存标注文件
- 使用ImageJ的自动保存功能
- 备份重要的标注结果

#### 高级功能

**批量处理**
```bash
# 使用ImageJ的批处理功能
# 可以编写宏脚本自动化处理
```

**标注验证**
- 检查mask质量
- 验证类别标签是否有效
- 确认标注覆盖度

**性能优化**
- 使用SSD存储提高文件读写速度
- 关闭不必要的应用程序释放内存
- 使用高分辨率显示器提高标注精度

#### 最佳实践

**标注流程**
1. **预览**: 先查看模型预测结果
2. **调整**: 微调不准确的边界
3. **添加**: 补充缺失的标注区域
4. **删除**: 移除错误的预测结果
5. **验证**: 检查标注质量
6. **保存**: 保存标注结果

**质量控制**
- 确保每个类别都有足够的标注样本
- 保持标注风格的一致性
- 定期与团队成员讨论标注标准

**效率提升**
- 建立标注模板和标准
- 使用快捷键和工具提高效率
- 定期总结和改进标注流程

### Pipeline 预训练模型权重使用指南

#### 概述
本指南说明如何在多阶段训练和微调pipeline中使用预训练模型权重。

#### 配置选项

**1. 全局预训练模型配置**

在两个pipeline配置文件中，都添加了以下配置项：

```python
# --- 预训练模型配置 ---
USE_PRETRAINED_MODEL = False  # 是否使用预训练模型
PRETRAINED_MODEL_PATH = Path("models/checkpoints/your_pretrained_model.pth")  # 预训练模型路径

# --- 模型权重保存配置 ---
MODEL_SAVE_DIR = Path("models/checkpoints/multistage_train")  # 模型权重保存目录
SAVE_BEST_MODEL = True  # 是否保存最佳模型
SAVE_LAST_MODEL = False  # 是否保存最后一个epoch的模型
SAVE_CHECKPOINT = True  # 是否保存训练检查点（包含优化器状态）
```

**2. 阶段级预训练模型配置**

每个阶段都可以单独配置预训练模型路径：

```python
STAGES = [
    {
        "name": "stage1",
        # ... 其他配置
        "pretrained_model_path": None,  # 可指定该阶段的预训练模型路径
    },
    {
        "name": "stage2",
        # ... 其他配置
        "pretrained_model_path": None,  # 自动使用前一阶段模型
    }
]
```

#### 使用方式

**1. 多阶段训练 Pipeline (multistage_train_pipeline.py)**

**第一阶段：**
- 如果 `USE_PRETRAINED_MODEL = True`，会尝试加载 `PRETRAINED_MODEL_PATH` 指定的模型
- 如果找不到预训练模型，使用随机初始化
- 如果 `USE_PRETRAINED_MODEL = False`，直接使用随机初始化

**后续阶段：**
- 自动使用前一阶段的最佳模型权重
- 如果找不到前一阶段模型，使用随机初始化

**2. 多阶段微调 Pipeline (multistage_finetune_pipeline.py)**

**第一阶段：**
- 如果 `USE_PRETRAINED_MODEL = True`，优先加载完整预训练模型
- 如果找不到完整预训练模型，回退到编码器预训练权重
- 如果 `USE_PRETRAINED_MODEL = False`，使用编码器预训练权重

**后续阶段：**
- 自动使用前一阶段的最佳模型权重
- 如果找不到前一阶段模型，使用随机初始化

#### 配置示例

**示例1：使用预训练模型进行多阶段训练**

```python
# configs/train/multistage_train_config.py
USE_PRETRAINED_MODEL = True
PRETRAINED_MODEL_PATH = Path("models/checkpoints/active_learning_best_model.pth")

# 自定义模型保存路径
MODEL_SAVE_DIR = Path("models/checkpoints/my_custom_training")
SAVE_BEST_MODEL = True
SAVE_LAST_MODEL = True
SAVE_CHECKPOINT = True

STAGES = [
    {
        "name": "stage1",
        # ... 其他配置
        "pretrained_model_path": Path("models/checkpoints/ssl_pretrained_model.pth"),  # 指定第一阶段模型
    },
    {
        "name": "stage2",
        # ... 其他配置
        "pretrained_model_path": None,  # 使用stage1的输出
    }
]
```

**示例2：使用预训练模型进行多阶段微调**

```python
# configs/finetune/multistage_finetune_config.py
USE_PRETRAINED_MODEL = True
PRETRAINED_MODEL_PATH = Path("models/checkpoints/active_learning_best_model.pth")

# 自定义模型保存路径
MODEL_SAVE_DIR = Path("models/checkpoints/my_custom_finetune")
SAVE_BEST_MODEL = True
SAVE_LAST_MODEL = False
SAVE_CHECKPOINT = True

STAGES = [
    {
        "name": "stage1",
        # ... 其他配置
        "pretrained_encoder_path": Path("models/checkpoints/ssl_pretrained_unet/best_ssl_encoder.pth"),
        "pretrained_model_path": Path("models/checkpoints/active_learning_best_model.pth"),  # 完整预训练模型
    },
    {
        "name": "stage2",
        # ... 其他配置
        "pretrained_model_path": None,  # 使用stage1的输出
    }
]
```

#### 权重加载策略

**支持的Checkpoint格式**

Pipeline会自动识别以下checkpoint格式：

1. **标准格式**：
   ```python
   {
       'model_state_dict': model.state_dict(),
       'epoch': epoch,
       'val_dice': val_dice,
       # ... 其他信息
   }
   ```

2. **简化格式**：
   ```python
   {
       'state_dict': model.state_dict(),
       # ... 其他信息
   }
   ```

3. **直接格式**：
   ```python
   model.state_dict()  # 直接是state_dict
   ```

**加载优先级**

1. **多阶段训练**：
   - 第一阶段：全局预训练模型 → 阶段预训练模型 → 随机初始化
   - 后续阶段：前一阶段模型 → 随机初始化

2. **多阶段微调**：
   - 第一阶段：完整预训练模型 → 编码器预训练权重 → 随机初始化
   - 后续阶段：前一阶段模型 → 随机初始化

#### 模型权重保存配置

**保存选项**

1. **最佳模型** (`SAVE_BEST_MODEL = True`)：
   - 保存验证集性能最好的模型
   - 文件名格式：`{stage_name}_best_model.pth`

2. **最后一个epoch模型** (`SAVE_LAST_MODEL = True`)：
   - 保存训练结束时的模型
   - 文件名格式：`{stage_name}_last_model.pth`

3. **训练检查点** (`SAVE_CHECKPOINT = True`)：
   - 保存包含优化器状态的完整检查点
   - 可用于断点续训
   - 文件名格式：`{stage_name}_checkpoint.pth`

**保存路径配置**

- **MODEL_SAVE_DIR**：指定模型权重保存的根目录
- 每个阶段的模型会保存在：`{MODEL_SAVE_DIR}/{stage_name}_*.pth`
- 目录不存在时会自动创建

#### 注意事项

1. **路径设置**：确保预训练模型路径正确且文件存在
2. **模型兼容性**：预训练模型应与当前模型架构兼容
3. **权重格式**：支持多种checkpoint格式，会自动识别
4. **回退机制**：如果预训练模型加载失败，会自动回退到随机初始化
5. **日志输出**：会显示详细的权重加载过程和结果
6. **存储空间**：根据保存选项，确保有足够的磁盘空间

#### 运行命令

```bash
# 运行多阶段训练
python -m pipelines.multistage_train_pipeline

# 运行多阶段微调
python -m pipelines.multistage_finetune_pipeline
```

#### 故障排除

**常见问题**

1. **模型路径不存在**：
   - 检查 `PRETRAINED_MODEL_PATH` 路径是否正确
   - 确认模型文件是否存在

2. **权重格式不兼容**：
   - 检查预训练模型的state_dict格式
   - 确认模型架构是否匹配

3. **CUDA内存不足**：
   - 减少batch_size
   - 使用CPU进行权重加载

**调试建议**

1. 启用详细日志输出
2. 检查模型文件大小和格式
3. 验证模型架构兼容性
4. 测试单个阶段的权重加载
