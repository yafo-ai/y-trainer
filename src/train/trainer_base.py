from abc import ABC, abstractmethod
from typing import List
from src.train.config import TrainConfig
from src.train.loss_logger import LossLogger


class TrainerBase(ABC):
    """训练器基类，定义训练器的基本接口"""

    def __init__(self, config:TrainConfig):
        self._config = config
        self.process_logger = LossLogger(self._config)
        

    # 加载数据
    @abstractmethod
    def load_dataset(self,tokenizer):
        raise NotImplementedError

    @abstractmethod
    def collate_fn(self, batch:List[dict],tokenizer):
        raise NotImplementedError
    
    # 数据训练
    @abstractmethod
    def train(self,model, optimizer, batcher,teacher_model=None):
        raise NotImplementedError