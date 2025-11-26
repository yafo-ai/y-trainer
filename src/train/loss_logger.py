from torch.utils.tensorboard import SummaryWriter
import matplotlib
import matplotlib.pyplot as plt
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import logging

from src.train.config import TrainConfig

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LossLogger:

    """
    使用Tensorboard工具收集记录训练日志
    """
    
    def __init__(self, config:TrainConfig):
        self._config = config
        self.writer = None
        self.logger_init()

    def logger_init(self): 
        if self._config.use_tensorboard:
            if self._config.local_rank == 0:
                self.writer = SummaryWriter(log_dir=self._config.tensorboard_path)

    def write_tb_log(self, epoch, step, value):
        if self._config.use_tensorboard:
            if self._config.local_rank == 0:
                self.writer.add_scalar(f'Loss/train ep {epoch}', value, step)

    def close(self):
        if self._config.use_tensorboard:
            if self._config.local_rank == 0:
                if self.writer != None:
                    self.export_logs_to_images()
                    self.writer.close()

    def export_logs_to_images(self):
        try:
            matplotlib.use('Agg')  
            event_acc = EventAccumulator(self._config.tensorboard_path)
            event_acc.Reload()
            
            output_dir=os.path.join(self._config.output_dir, "tensorboard_logs")

            os.makedirs(output_dir, exist_ok=True)

            for tag in event_acc.Tags()["scalars"]:
                events = event_acc.Scalars(tag)
                steps = [e.step for e in events]
                values = [e.value for e in events]

                plt.figure(figsize=(10, 6))
                
                plt.plot(steps, values, marker='o', markersize=2, linestyle='-')
                plt.title(tag)
                plt.xlabel("Steps")
                plt.ylabel("Loss")
                plt.grid(True)
                
                safe_tag = tag.replace("/", "_")
                plt.savefig(os.path.join(output_dir, f"{safe_tag}.png"), bbox_inches="tight")
                plt.close()
                
            print(f"save tensorboard logs to {output_dir}")
            logger.info(f"save tensorboard logs to {output_dir}")
        except Exception as e:
            print(f"Fail to save logs pic: {str(e)}")
            logger.error(f"Fail to save logs pic: {str(e)}")

                