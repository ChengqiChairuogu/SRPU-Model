import importlib
import os
import datetime

class Logger:
    def __init__(self, config):
        self.logger_type = config.get("logger", "wandb")
        self.project = config.get("project", "SRPU-Model")
        self.log_dir = config.get("log_dir", "./runs")
        self._logger = None
        # 生成时间戳
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        if self.logger_type == "wandb":
            try:
                from configs import wandb_config as cfg_wandb
                os.environ["WANDB_MODE"] = cfg_wandb.WANDB_MODE
                print(f"Wandb模式设置为: {cfg_wandb.WANDB_MODE}")
                if "ssl" in self.project.lower():
                    project_name = cfg_wandb.PROJECT_NAME_SSL
                else:
                    project_name = cfg_wandb.PROJECT_NAME_SUPERVISED
                wandb = importlib.import_module("wandb")
                run_name = f"{project_name}_{timestamp}"
                # 使用配置的log_dir作为wandb的dir参数
                wandb.init(project=project_name, config=config, name=run_name, dir=self.log_dir)
                self._logger = wandb
                print(f"Wandb初始化完成，项目: {project_name}, run_name: {run_name}, 日志目录: {self.log_dir}")
            except ImportError as e:
                print(f"警告: 无法导入wandb_config，使用默认配置: {e}")
                wandb = importlib.import_module("wandb")
                run_name = f"{self.project}_{timestamp}"
                # 使用配置的log_dir作为wandb的dir参数
                wandb.init(project=self.project, config=config, name=run_name, dir=self.log_dir)
                self._logger = wandb
        elif self.logger_type == "tensorboard":
            tb = importlib.import_module("torch.utils.tensorboard")
            log_dir_with_time = f"{self.log_dir}_{timestamp}"
            self._logger = tb.SummaryWriter(log_dir=log_dir_with_time)
            print(f"Tensorboard日志目录: {log_dir_with_time}")
        else:
            raise ValueError("logger must be 'wandb' or 'tensorboard'")

    def log(self, metrics, step=None):
        if self._logger is None:
            return
        if self.logger_type == "wandb":
            self._logger.log(metrics, step=step)
        elif self.logger_type == "tensorboard":
            for k, v in metrics.items():
                if isinstance(v, (int, float)) or (hasattr(v, 'dtype') and str(v.dtype).startswith('float')):
                    self._logger.add_scalar(k, v, step)

    def close(self):
        if self._logger is None:
            return
        if self.logger_type == "tensorboard":
            self._logger.close()
        elif self.logger_type == "wandb":
            self._logger.finish() 