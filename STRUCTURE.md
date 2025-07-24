# SRPU-Model 目录结构与详细说明

本文件详细介绍 SRPU-Model 项目各目录及其子文件/子模块的功能，便于新用户和团队成员快速理解和扩展。

---

SRPU-Model/
├── configs/                     # 配置文件目录
│   ├── base.py                  # 全局基础配置（路径、类别、图像尺寸等）
│   ├── augmentation_config.py   # 数据增强参数配置
│   ├── dataset_config.py        # 数据集划分比例、随机种子等
│   ├── json_config.py           # json索引生成相关配置
│   ├── wandb_config.py          # wandb日志系统全局配置
│   ├── finetune/
│   │   ├── finetune_config.py           # 单阶段微调参数
│   │   └── multistage_finetune_config.py# 多阶段微调参数
│   ├── train/
│   │   ├── train_config.py              # 单阶段有监督训练参数
│   │   └── multistage_train_config.py   # 多阶段有监督训练参数
│   ├── selfup/
│   │   ├── ssl_config.py                # 自监督预训练参数
│   │   └── ssl_inspect_config.py        # 自监督模型评估参数
│   └── inference/
│       ├── inference_config.py           # 推理参数
│       └── inspection_config.py          # 推理结果分析参数
│
├── data/                        # 数据目录
│   ├── raw/                     # 原始未标注SEM图像，按实验/批次分类
│   ├── labeled/                 # 已标注的图像及掩码，文件名与raw一致
│   ├── inference/
│   │   ├── input/               # 推理输入图片
│   │   └── output/              # 推理输出掩码
│
├── datasets/                    # 数据集定义与原始/掩码图片
│   ├── sem_datasets.py          # 主分割数据集类，支持多数据集联合、分层采样
│   ├── ssl_dataset.py           # 自监督/半监督任务专用数据集类
│   ├── stratified_sampler.py    # 分层采样器实现
│   ├── dataset1_LInCl/
│   │   ├── raw_images/          # LInCl原始图片
│   │   └── masks_3class/        # LInCl三分类掩码
│   ├── dataset2_LPSCl/
│   │   ├── raw_images/          # LPSCl原始图片
│   │   └── masks_3class/        # LPSCl三分类掩码
│   └── dataset3_LNOCl/
│       ├── raw_images/          # LNOCl原始图片
│       └── masks_3class/        # LNOCl三分类掩码
│
├── json/                        # 数据集索引json文件
│   ├── master_labeled_dataset.json      # 全部标注样本索引
│   ├── master_ssl_dataset.json          # 全部无标注样本索引
│   ├── dataset*_*.json                  # 各子数据集索引
│   └── ...                              # 其他json索引
│
├── models/                      # 模型结构与权重
│   ├── encoders/
│   │   ├── unet_encoder.py      # U-Net编码器
│   │   ├── resnet_encoder.py    # ResNet编码器
│   │   ├── efficientnet_encoder.py # EfficientNet编码器
│   │   └── dinov2_encoder.py    # DINOv2编码器
│   ├── decoders/
│   │   ├── unet_decoder.py      # U-Net解码器
│   │   └── deeplab_decoder.py   # DeepLabV3+解码器
│   ├── segmentation_unet.py     # U-Net分割模型主结构
│   ├── unet_autoencoder.py      # U-Net自编码器结构（自监督用）
│   ├── mae_model.py             # MAE自监督模型结构
│   └── checkpoints/
│       └── ssl_pretrained_unet/ # 各类模型权重保存目录
│
├── tasks/                       # 任务脚本
│   ├── train_task.py                    # 单阶段有监督训练主入口
│   ├── finetune_task.py                 # 单阶段微调主入口
│   ├── ssl_pretrain_task.py             # 自监督预训练主入口
│   ├── ssl_inspect_task.py              # 自监督模型评估脚本
│   ├── inference_task.py                # 推理脚本
│   ├── inspect_validation_results.py    # 验证集结果分析与可视化
│   └── __init__.py                      # 包初始化
│
├── pipelines/                   # 多阶段训练/微调主入口脚本
│   ├── multistage_train_pipeline.py     # 多阶段有监督训练主入口
│   └── multistage_finetune_pipeline.py  # 多阶段微调主入口
│
├── utils/                       # 工具函数
│   ├── augmentation.py                  # 数据增强实现
│   ├── dataset_statistics_calculator.py # 统计数据集分布、类别比例
│   ├── diagnose_image_sizes.py          # 检查图片尺寸一致性
│   ├── json_generator.py                # 自动生成/更新json索引文件
│   ├── logger.py                        # 日志系统适配（wandb/tensorboard），自动管理日志目录和run name
│   └── training_utils.py                # 训练通用工具（损失函数、评估、训练循环等）
│
├── tensorboard/                 # tensorboard日志目录
├── wandb/                       # wandb日志目录（如存在）
├── .ipynb_checkpoints/          # Jupyter Notebook检查点
├── environment.yml              # Conda 环境依赖文件
├── README.md                    # 项目说明文档
└── STRUCTURE.md                 # 当前项目结构描述文件