self_segmentation/             # ── 根目录
│
├── configs/                   # 配置文件目录
│   ├── active/                # 活动配置文件
│   ├── finetune/              # 微调配置文件
│   ├── selfsup/               # 自我监督学习配置文件
│   ├── semiup/                # 半监督学习配置文件
│   └── base.py                # 基础配置文件
│
├── data/                      # 数据存储目录
│   ├── labeled/               # 已标注数据
│   ├── pseudo/                # 伪标签数据
│   └── raw/                   # 原始数据
│
├── datasets/                  # 数据集处理与加载相关脚本
├── json/                      # JSON 配置文件或数据
├── logs/                      # 日志文件目录
│
├── models/                    # 模型定义目录
│   ├── checkpoints/           # 模型权重保存点
│   ├── decoders/              # 解码器模块
│   └── encoders/              # 编码器模块
│
├── tasks/                     # 特定任务脚本或模块
├── utils/                     # 工具函数模块
│
├── environment.yml            # Conda 环境配置文件
├── pipeline.py                # 主要处理流程或流水线脚本
├── README.md                  # 项目说明文件
└── STRUCTURE.md               # 项目结构描述文件