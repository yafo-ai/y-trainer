from torch.utils.tensorboard import SummaryWriter
import matplotlib
import matplotlib.pyplot as plt
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProcessLogger:
    
    def __init__(self, training_config, model_config):
        self.training_config = training_config
        self.model_config = model_config
        self.writer = None
        self.logger_init()

    def logger_init(self): 
        if self.training_config.use_tensorboard:
            if self.training_config.local_rank == 0:
                self.writer = SummaryWriter(log_dir=self.training_config.tensorboard_path)

    def write_tb_log(self, epoch, step, value):
        if self.training_config.use_tensorboard:
            if self.training_config.local_rank == 0:
                self.writer.add_scalar(f'Loss/train ep {epoch}', value, step)

    def close(self):
        if self.training_config.use_tensorboard:
            if self.training_config.local_rank == 0:
                if self.writer != None:
                    export_tensorboard_logs_to_images(self.training_config.tensorboard_path, os.path.join(self.training_config.output_dir, "tensorboard_logs"))
                    self.writer.close()
                    

def export_tensorboard_logs_to_images(log_dir, output_dir) -> None:
    try:
        matplotlib.use('Agg')  
        event_acc = EventAccumulator(log_dir)
        event_acc.Reload()
        

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
    except Exception as e:
        print(f"Fail to save logs pic: {str(e)}")