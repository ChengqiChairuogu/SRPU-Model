# configs/dataset_config.py
# 数据集划分相关参数，供train/finetune等任务统一调用

SPLIT_RATIO = (0.8, 0.1, 0.1)  # 训练/验证/测试集比例
SPLIT_SEED = 42  # 随机种子，保证划分可复现 