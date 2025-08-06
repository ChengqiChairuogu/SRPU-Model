from easydict import EasyDict as edict
import os
from pathlib import Path

def get_evaluation_config():
    """
    获取模型评估任务的统一配置。
    合并了原有的 inspection_config 和 difference_config 功能。
    """
    config = edict()

    # =================================================================================
    #                           !!! 用户需要修改的参数 !!!
    # =================================================================================
    
    # 1. 指向你想要评估的、已经训练好的模型权重文件 (.pth)
    # TODO: 请确保此路径是正确的！
    config.CHECKPOINT_PATH = "/cluster/home/cheng_qi/SRPU-Model/models/checkpoints/sem_segmentation_from_scratch_unet_unet/best_model.pth"

    # 2. 保存可视化对比图和评估结果的输出目录
    config.OUTPUT_DIR = "./evaluation_results"

    # 3. 你希望评估的图片数量。设置为 -1 将评估验证集中的所有图片。
    config.NUM_IMAGES_TO_INSPECT = 50

    # 4. 确认用于训练该模型的模型架构是什么
    #    这必须与训练时使用的模型完全一致！
    config.MODEL_CONFIG = edict()
    config.MODEL_CONFIG['encoder_name'] = 'unet_encoder' 
    config.MODEL_CONFIG['decoder_name'] = 'unet_decoder'

    # =================================================================================
    #                              数据加载器配置
    # =================================================================================
    config.DATA_LOADER_CONFIG = edict()
    config.DATA_LOADER_CONFIG['val'] = edict()
    
    # [关键修正]
    # 使用 `file_list` 作为参数名，以匹配您的 SemSegmentationDataset 类的初始化函数。
    # 路径指向您用于验证的json文件。
    config.DATA_LOADER_CONFIG['val']['file_list'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../json/master_labeled_dataset_val.json'))
    
    # 以下参数一般无需修改
    config.DATA_LOADER_CONFIG['val']['batch_size'] = 1      # 评估时必须为1
    config.DATA_LOADER_CONFIG['val']['num_workers'] = 2

    # =================================================================================
    #                              评估指标配置
    # =================================================================================
    
    # 支持的评估指标
    config.EVAL_METRICS = ['dice', 'iou']
    
    # 评估结果保存路径（可选，None则只打印不保存）
    config.RESULT_SAVE_PATH = None  # 如 'evaluation_results/evaluation_results.txt'
    
    # 预测mask目录（用于外部预测结果的评估）
    config.PRED_MASK_DIR = Path('data/inference/output')
    
    # GT mask目录（如用json则可留空）
    config.GT_MASK_DIR = None
    
    # 可选：指定多个json文件，自动查找GT
    config.GT_JSON_PATHS = [
        Path('json/dataset1_LInCl.json'),
        Path('json/dataset2_LPSCl.json'),
        Path('json/dataset3_LNOCl.json'),
    ]

    return config 