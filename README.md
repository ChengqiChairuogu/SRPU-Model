# SRPU-Model 项目说明

## 项目简介

SRPU-Model 是一个面向扫描电镜（SEM）图像分割的自进化深度学习平台，集成了**监督学习**、**自监督学习**、**半监督学习**和**主动学习**等多种策略，目标是用有限标注数据和大量无标注数据，持续提升分割模型性能，实现“自我成长”的智能分割系统。

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
├── utils/           # 工具函数与脚本
├── runs/            # 日志主目录（下分 tensorboard/ 和 wandb/）
│   ├── tensorboard/ # tensorboard日志（每次实验自动新建子目录）
│   └── wandb/       # wandb日志（每次实验自动新建子目录）
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
  - `inspection_config.py`：推理结果分析参数。

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
- **stratified_sampler.py**：分层采样器实现，保证各子数据集均衡采样。

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
- **inspect_validation_results.py**：验证集结果分析与可视化。
- **multistage_train_task.py**：多阶段有监督训练主入口。
- **multistage_finetune_task.py**：多阶段微调主入口。

### utils/ 工具函数
- **augmentation.py**：数据增强实现。
- **dataset_statistics_calculator.py**：统计数据集分布、类别比例。
- **diagnose_image_sizes.py**：检查图片尺寸一致性。
- **json_generator.py**：自动生成/更新json索引文件。
- **logger.py**：日志系统适配（wandb/tensorboard），自动管理日志目录和run name。
- **training_utils.py**：训练通用工具（损失函数、评估、训练循环等）。

### 其它重要文件
- **environment.yml**：Conda环境依赖说明。
- **README.md**：项目说明文档。
- **STRUCTURE.md**：代码结构和扩展说明。
- **pipeline.py**：（预留）可用于统一调度多阶段流程的主入口。

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
python utils/json_generator.py --mode generate_all
```

---

## 日志系统与实验管理

### 日志系统选择
- 在各类 config 文件（如 `configs/train/train_config.py`、`configs/finetune/finetune_config.py`、`configs/selfup/ssl_config.py`、多阶段config等）中通过 `LOGGER` 字段选择日志方式：

  ```python
  LOGGER = "wandb"  # 可选 "wandb" 或 "tensorboard"
  ```

- **日志目录结构已标准化为：**
  - `runs/tensorboard/任务名_时间戳`  （如 tensorboard 日志）
  - `runs/wandb/任务名_时间戳`        （如 wandb 日志）
  - 每次实验会自动在 runs/tensorboard 或 runs/wandb 下新建唯一子目录，便于管理和查找。

### 日志查看
- **Wandb**：支持在线/离线模式，详见 [wandb 官网](https://wandb.ai/)。
- **Tensorboard**：如选择 tensorboard，运行：

  ```bash
  tensorboard --logdir runs/tensorboard
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

### 多阶段训练/微调
- **多阶段有监督训练**  
  配置：`configs/train/multistage_train_config.py`  
  运行：  
  ```bash
  python tasks/multistage_train_task.py
  ```
- **多阶段微调**  
  配置：`configs/finetune/multistage_finetune_config.py`  
  运行：  
  ```bash
  python tasks/multistage_finetune_task.py
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

---

如需详细使用说明、参数解释或遇到问题，欢迎随时提问！

---

如果你有特殊硬件/平台需求、想集成新模型或有大规模实验需求，也欢迎联系开发者团队。
