from argparse import ArgumentParser, Namespace
from dataclasses import asdict, dataclass
import os
import json
import logging
from enum import Enum
from typing import List
import torch
# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# 定义训练类型枚举
class TrainingType(str, Enum):
    CPT = "cpt"
    SFT = "sft"



def parse_data_type(data_type_str: str) -> torch.dtype:

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    
    if data_type_str not in dtype_map:
        raise ValueError(f"Not supported data type: {data_type_str}")
    
    return dtype_map[data_type_str]


@dataclass
class TrainConfig:

    """训练参数类"""

    # 模型路径
    model_path_to_load: str=""

    # 是否使用LoRA
    use_lora: bool = False

    # 数据类型
    data_type: torch.dtype = torch.bfloat16

    #系统提示词
    system_prompt: str = "You are a helpful assistant."

    # 是否使用DeepSpeed
    use_deepspeed: bool = False

    # DeepSpeed json 配置路径
    deepspeed_cfg_path: str = ""

    # 是否使用梯度检查点
    gradit_checkpoing: bool = False
    

    """LoraConfig配置"""

    #LoRA权重的文件路径
    lora_path: str = ""

    #rank 越高，LoRA适配器能够学习到的权重更新越复杂，可能带来更好的微调效果
    lora_rank: int = 16

    # 适应强度：alpha 越大，LoRA适配器的贡献越大，模型更倾向于学习新任务的模式；alpha 越小，模型更依赖原始预训练权重。
    lora_alpha: float = 32

    # 防止过拟合：较高的 dropout（如0.2-0.5）可以增强模型的泛化能力，尤其适用于数据量较小或任务复杂的场景。
    lora_dropout: float = 0.2

    # 需要应用 LoRA 的目标模块列表
    lora_target_modules: List[str] = None


    """训练配置数据类"""

    # 训练轮数
    epoch: int=3

    # 学习率
    lr: float=5e-6

    # 批次大小
    batch_size: int=1

    # 梯度累积步数
    gradient_accumulation_steps: int=1

    # 训练数据路径
    data_path: str=""

    # 输出目录
    output_dir: str=""

    # 随机种子
    seed: int = 42

    # 是否使用TensorBoard
    use_tensorboard: bool = False

    # TensorBoard路径
    tensorboard_path: str = ""

    # 是否使用进度条
    use_process_bar: bool = False

    # 训练类型
    training_type: str = TrainingType.SFT.value

    # 检查点轮次列表
    checkpoint_epoch: List[int] = None

    # 本地进程排名
    local_rank: int = 0

    # 世界大小（总进程数/GPU数量）
    world_size: int = 1
    

    # 核心算法
    use_nlirg: bool = True

    """训练特定配置"""

    # token批次大小
    token_batch: int = 10

    # 蒸馏
    distillition: bool = False

    # 教师模型路径
    teacher_model_path: str = ""

    # 原始损失系数
    coefficient_of_origin_loss: float = 0.5
    
    max_seq_len: int = 20520

    """CPT训练特定配置"""
    pack_length: int = -1

    """RL训练特定配置"""
    rl_type: str = "agent"


    def __post_init__(self):
        # if self.lora_target_modules is None or len(self.lora_target_modules) == 0:
        #     self.lora_target_modules = ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

        # 验证对象属性
        self._validate()

        #保存训练配置
        self._save()


    def _validate(self) -> None:
        """验证模型配置"""
        if not os.path.exists(self.model_path_to_load):
            logger.warning(f"Model path does not exist: {self.model_path_to_load}")
            raise Exception(f"Model path does not exist: {self.model_path_to_load}")
        if not any(os.listdir(self.model_path_to_load)):
            logger.warning(f"Model path is empty: {self.model_path_to_load}")
            raise Exception(f"Model path is empty，please save model to->{self.model_path_to_load}")
    
        if not os.path.exists(self.data_path):
            logger.warning(f"Data path does not exist->{self.data_path}")
            raise Exception(f"Data path does not exist: {self.data_path}")  
        
        if not os.path.exists(self.output_dir):
            logger.info(f"Creating output directory: {self.output_dir}")
            os.makedirs(self.output_dir, exist_ok=True)
        elif self.local_rank==0 and  any(os.listdir(self.output_dir)):
            raise Exception(f"output_dir must be empty: {self.output_dir}")
        
        if self.epoch <= 0:
            raise ValueError(f"Epoch must be positive, got {self.epoch}")
        
        if self.lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {self.lr}")
        
        if self.batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {self.batch_size}")
        
        if self.gradient_accumulation_steps <= 0:
            raise ValueError(f"Gradient accumulation steps must be positive, got {self.gradient_accumulation_steps}")
        
        if self.use_lora:
            if self.lora_rank <= 0:
                raise ValueError(f"LoRA rank must be positive, got {self.lora_rank}")
            if self.lora_alpha <= 0:
                raise ValueError(f"LoRA alpha must be positive, got {self.lora_alpha}")
            if not 0 <= self.lora_dropout <= 1:
                raise ValueError(f"LoRA dropout must be between 0 and 1, got {self.lora_dropout}")
            if self.lora_target_modules is None or len(self.lora_target_modules) == 0:
                raise ValueError("LoRA target modules cannot be empty when use_lora is True")
            # valid_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            # if not all(module in valid_modules for module in self.lora_target_modules):
            #     raise ValueError("LoRA target modules must be a subset of ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']")
            
    def _save(self) -> None:
        """
        保存训练配置到JSON文件
        """
        try:
            if self.local_rank == 0:

                config_dir = os.path.join(self.output_dir, "configs")

                os.makedirs(config_dir, exist_ok=True)
                    
                config_path = os.path.join(config_dir, "training_config.json")
                
                data = asdict(self)
                data['data_type'] = str(self.data_type)  # 转为字符串
    
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to save config to file: {str(e)}")
            raise
    

