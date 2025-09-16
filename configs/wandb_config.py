# self_segmentation/configs/wandb_config.py

# --- Wandb 主模式设置 ---
# "online": (默认) 将数据实时同步到 wandb.ai 服务器。需要网络和登录。
# "offline": 在本地磁盘上保存 wandb 运行数据，适用于无网络环境。默认为此模式。
# "disabled": 完全禁用 wandb，不记录任何内容。
WANDB_MODE = "offline"

# --- 项目名称 ---
# 用于监督学习和微调任务的项目名称
PROJECT_NAME_SUPERVISED = "SRPU-Model-Supervised"

# 用于自监督学习任务的项目名称
PROJECT_NAME_SSL = "SRPU-Model-SSL"


# --- 使用说明 ---
# 1. 默认情况下，所有运行都将在本地的 `wandb` 目录中创建离线数据。
# 2. 如果您希望在线同步，只需将 WANDB_MODE 的值更改为 "online"。
# 3. 在有网络的环境下，您可以使用命令行工具同步离线运行：
#    在您的项目根目录下运行: wandb sync wandb/offline-run-xxxxxxxx-xxxxxx