self_segmentation/
├── configs/                # 存放所有任务和模型的配置文件
│   ├── active/             # 主动学习 (Active Learning) 相关配置 (未来规划)
│   ├── finetune/           # 模型微调 (Fine-tuning) 相关配置
│   ├── selfup/             # 自监督学习 (Self-supervised Learning) 相关配置
│   ├── semiup/             # 半监督学习 (Semi-supervised Learning) 相关配置 (未来规划)
│   └── base.py             # 基础/通用配置Python脚本 (替代base.yaml)
│
├── data/                   # 存放各类数据集及相关元数据
│   ├── labeled/            # 存放人工高质量标注的掩码文件 (图像本身在raw中)
│   ├── pseudo/             # 存放由模型生成的伪标签掩码文件 (未来规划)
│   └── unlabeled/          # (逻辑概念) 指向 raw/ 中未被 labeled/ 或 pseudo/ 引用的数据，或单独存放特定于无监督任务的图像
│
├── datasets/               # 存放处理后的、可直接供模型使用的数据集文件 (例如PyTorch Dataset脚本)
├── logs/                   # 存放训练过程中的日志文件 (例如TensorBoard日志)
├── json/                   # 存放描述数据集结构和元数据的JSON文件 (例如，用于3D数据堆叠的序列信息)
│
├── models/                 # 存放模型架构、组件和训练好的权重
│   ├── checkpoints/        # 保存训练过程中的模型权重/检查点
│   ├── decoders/           # 分割模型的解码器模块定义
│   └── encoders/           # 分割模型的编码器模块定义 (可包含预训练backbone)
│
├── tasks/                  # 存放不同学习任务的执行脚本和主要逻辑
├── utils/                  # 通用工具函数、辅助脚本
├── environment.yml         # Conda 环境配置文件
├── pipeline.py             # (未来规划) 整个项目工作流的编排/总控脚本
├── README.md               # 项目说明、设置指南和使用方法
└── STRUCTURE.md            # 本文件，描述项目的文件和目录组织结构
