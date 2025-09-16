from easydict import EasyDict as edict
import os
from pathlib import Path

def get_evaluation_config():
    """
    获取模型评估任务的统一配置。
    支持两种评估模式：
    1. 随机抽取数据集中的图片进行评估
    2. 评估推理预测出的图像
    """
    config = edict()

    # =================================================================================
    #                           !!! 用户需要修改的参数 !!!
    # =================================================================================
    
    # 1. 模型类型选择
    # 可选值: 'supervised' (监督学习模型)
    config.MODEL_TYPE = 'supervised'  
    
    # 2. 指向你想要评估的、已经训练好的模型权重文件 (.pth)
    # TODO: 请确保此路径是正确的！
    config.CHECKPOINT_PATH = "models/checkpoints/multistage_finetune/stage2_best_model.pth"

    # 3. 保存可视化对比图和评估结果的输出目录
    config.OUTPUT_DIR = "data/evaluation_results"
    
    # 4. 评估模式选择
    # 可选值: 'random_sample' (随机抽取) 或 'inference_result' (推理结果)
    config.EVALUATION_MODE = 'random_sample'
    
    # 5. 模型架构配置
    #    这必须与训练时使用的模型完全一致！
    config.MODEL_CONFIG = edict()
    
    # 监督学习模型配置
    config.MODEL_CONFIG['encoder_name'] = 'unet' 
    config.MODEL_CONFIG['decoder_name'] = 'unet'
    config.MODEL_CONFIG['model_class'] = 'SegmentationUNet'

    # =================================================================================
    #                              模式1: 随机抽取评估配置
    # =================================================================================
    if config.EVALUATION_MODE == 'random_sample':
        # 你希望评估的图片数量。设置为 -1 将评估验证集中的所有图片。
        # 注意：如果验证集样本不足，建议设置为 -1 或较小的数值
        config.NUM_IMAGES_TO_INSPECT = -1  # 改为-1，评估验证集中的所有图片
        
        # 数据加载器配置
        config.DATA_LOADER_CONFIG = edict()
        config.DATA_LOADER_CONFIG['val'] = edict()
        
        # 使用 `json_file_identifier` 作为参数名，以匹配 SemSegmentationDataset 类的初始化函数。
        # 路径指向您用于验证的json文件。
        config.DATA_LOADER_CONFIG['val']['json_file_identifier'] = "master_sharpness_averaged_dataset.json"
        
        # 以下参数一般无需修改
        config.DATA_LOADER_CONFIG['val']['batch_size'] = 1      # 评估时必须为1
        config.DATA_LOADER_CONFIG['val']['num_workers'] = 2
        
        # 随机种子设置（确保结果可重现）
        config.RANDOM_SEED = 42
        
        # 新增：是否同时评估训练集和验证集（用于全面性能评估）
        config.EVALUATE_BOTH_SPLITS = True
        config.DATA_LOADER_CONFIG['train'] = edict()
        config.DATA_LOADER_CONFIG['train']['json_file_identifier'] = "master_sharpness_averaged_dataset.json"
        config.DATA_LOADER_CONFIG['train']['batch_size'] = 1
        config.DATA_LOADER_CONFIG['train']['num_workers'] = 2

    # =================================================================================
    #                              模式2: 推理结果评估配置
    # =================================================================================
    elif config.EVALUATION_MODE == 'inference_result':
        # 推理结果目录（包含模型预测的掩码图像）
        config.PRED_MASK_DIR = Path('data/inference/output')
        
        # 真实标签目录（包含对应的GT掩码）
        config.GT_MASK_DIR = Path('data/inference/gt_masks')
        
        # 或者使用JSON文件自动查找GT（推荐）
        config.GT_JSON_PATHS = [
            Path('json/dataset1_LInCl.json'),
            Path('json/dataset2_LPSCl.json'),
            Path('json/dataset3_LNOCl.json'),
        ]
        
        # 推理结果的文件扩展名
        config.PRED_MASK_EXTENSION = '_mask.png'
        
        # 是否只评估有对应GT的推理结果
        config.ONLY_EVALUATE_WITH_GT = True
        
        # 最大评估图片数量（-1表示评估所有）
        config.MAX_EVALUATION_IMAGES = -1

    # =================================================================================
    #                              通用评估配置
    # =================================================================================
    
    # 支持的评估指标
    config.EVAL_METRICS = ['dice', 'iou']
    
    # 评估结果保存路径（可选，None则只打印不保存）
    config.RESULT_SAVE_PATH = "data/evaluation_results/evaluation_results.txt"  # 如 'evaluation_results/evaluation_results.txt'
    
    # 可视化配置
    config.VISUALIZATION = edict()
    config.VISUALIZATION['save_images'] = True           # 是否保存可视化图像
    config.VISUALIZATION['image_format'] = 'png'         # 图像格式
    config.VISUALIZATION['dpi'] = 150                    # 图像分辨率
    config.VISUALIZATION['figure_size'] = (18, 6)        # 图像尺寸 (宽, 高)
    
    # 指标计算配置
    config.METRICS_CONFIG = edict()
    config.METRICS_CONFIG['dice_smooth'] = 1e-6          # Dice计算的平滑因子
    config.METRICS_CONFIG['iou_smooth'] = 1e-6           # IoU计算的平滑因子
    config.METRICS_CONFIG['ignore_background'] = False   # 是否忽略背景类别
    


    return config 