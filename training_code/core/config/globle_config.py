from argparse import ArgumentParser, Namespace
import os
import json
import logging
from enum import Enum
from typing import Dict, Optional, Tuple, Union, Any
from abc import ABC
import datetime
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


class BaseConfig(ABC):


    def __init__(self, config_dict: dict):
        self._config = config_dict
        
    def __getattr__(self, name):
        if name in self._config:
            return self._config[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def __setitem__(self, key: str, value: Any):
        self._config[key] = value

    def __contains__(self, key) -> bool:
        return key in self._config
    
    def __getitem__(self, key: str) -> Any:
        return self._config[key]


    def update(self, config_dict: dict):
        """ 
        Batch updating the config
        
        Args: config_dict (dict): Dict of config to be updated.
        """
        self._config.update(config_dict)
        
    def save(self, file_path: str) -> None:
        """
        Save the config to a JSON file
        
        Args:
            file_path: File path to save the config
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to save config to file: {str(e)}")
            raise
    
    @classmethod
    def load(cls, file_path: str) -> 'BaseConfig':
        """
        Load the config from a JSON file
        
        Args:
            file_path: JSON Path
            
        Returns:
            BaseConfig
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return cls(data)
        except Exception as e:
            logger.error(f"加载配置失败: {str(e)}")
            raise
    
    def to_dict(self):
        config_dict = {}
        for name in self._config:
            if isinstance(self._config[name], BaseConfig):
                config_dict[name] = self._config[name].to_dict()
            else:
                config_dict[name] = self._config[name]
        return config_dict



class LoraConfig(BaseConfig):
    """训练配置类"""
    def __init__(self, config_dict: dict):
        super().__init__(config_dict)




class ModelConfig(BaseConfig):
    def __init__(self, config_dict: dict):
        super().__init__(config_dict)



class TrainingConfig(BaseConfig):
    def __init__(self, config_dict: dict):
        super().__init__(config_dict)


def parse_args() -> Namespace:

    parser = ArgumentParser(description="Training configuration parser")
    
    # model config
    parser.add_argument("--model_path_to_load", type=str, required=True, help="Path to the model")

    parser.add_argument("--data_type", type=str, default="bfloat16", 
                        help="Data type for training, float16 or float32", 
                        choices=["float16", "float32", "bfloat16"])
    
    parser.add_argument("--system_prompt", type=str, default="You are a helpful assistant.", help="System prompt")

    # LoRA config
    parser.add_argument("--use_lora", action="store_true", help="Whether to use LoRA")
    parser.add_argument("--lora_path", type=str, default="", help="Path to trained LoRA model")
    parser.add_argument("--lora_rank", type=int, default=16, help="Rank of the LoRA model")
    parser.add_argument("--lora_alpha", type=float, default=32, help="Alpha parameter for LoRA")
    parser.add_argument("--lora_dropout", type=float, default=0.2, help="Dropout rate for LoRA")
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,v_proj,k_proj,o_proj", 
                        help="Target modules for LoRA, split by comma, do not type space there")

    # training config
    parser.add_argument("--training_type", type=str, default='sft', 
                        choices=[t.value for t in TrainingType], 
                        help="Training type: cpt, sft")

    parser.add_argument("--epoch", type=int, default=3, help="Epoch for training")
    parser.add_argument("--checkpoint_epoch", type=str, default="", help="Epoch for checkpoint")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate for training")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, 
                        help="Gradient accumulation steps for training")

    parser.add_argument('--save_optimizer_state', action="store_true", help='whether to save optimizer state')
    parser.add_argument('--use_deepspeed', action="store_true", help='whether to use deepspeed config')
    parser.add_argument('--deepspeed_cfg_path', type=str, default="", help='deepspeed config path, if not set, will automaticaly generate one with stage 3')
    parser.add_argument('--local_rank', type=int, default=0, help="Rank for current process")
    parser.add_argument('--enable_gradit_checkpoing', action="store_true", help='whether to set gradit checkpoing')

    # cpt config
    parser.add_argument("--pack_length", type=int, default=-1, 
                        help="Pack length for training, -1 means no pack")

    # sft config
    parser.add_argument("--use_NLIRG", action="store_true", help="Whether to use NLIRG")
    parser.add_argument("--max_seq_len", type=int, default=20520, help="Max sequence length for training")
    parser.add_argument("--token_batch", type=int, default=-1, 
                        help="Token batch for training, 0 means no token batch")
    parser.add_argument('--Distillition', action="store_true", help="Whether to use Distillition")
    parser.add_argument('--coefficient_of_origin_loss', type=float, default=0.5, help="Coefficient of origin loss")
    parser.add_argument('--teacher_model_path', type=str, default="", help="Path to the teacher model")

    # data paths
    parser.add_argument("--data_path", type=str, required=True, help="Path to the training data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")

    # logging config
    parser.add_argument("--use_tensorboard", action="store_true", help="Whether to use tensorboard")
    parser.add_argument("--tensorboard_path", type=str, default="", 
                        help="User can specify the path to tensorboard, default is the output_dir")
    
    parser.add_argument("--use_process_bar", action="store_true", help="Whether to use process bar")
    args = parser.parse_args()
    
    _validate_args(args)
    
    return args

def _validate_args(args: Namespace) -> None:

    if not os.path.exists(args.model_path_to_load):
        logger.warning(f"Model path does not exist: {args.model_path_to_load}")
    
    if not os.path.exists(args.data_path):
        logger.warning(f"Data path does not exist: {args.data_path}")
    
    if not os.path.exists(args.output_dir):
        logger.info(f"Creating output directory: {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)
    
    if args.epoch <= 0:
        raise ValueError(f"Epoch must be positive, got {args.epoch}")
    
    if args.lr <= 0:
        raise ValueError(f"Learning rate must be positive, got {args.lr}")
    
    if args.batch_size <= 0:
        raise ValueError(f"Batch size must be positive, got {args.batch_size}")
    
    if args.gradient_accumulation_steps <= 0:
        raise ValueError(f"Gradient accumulation steps must be positive, got {args.gradient_accumulation_steps}")
    
    if args.use_lora:
        if args.lora_rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {args.lora_rank}")
        
        if args.lora_alpha <= 0:
            raise ValueError(f"LoRA alpha must be positive, got {args.lora_alpha}")
        
        if args.lora_dropout < 0 or args.lora_dropout > 1:
            raise ValueError(f"LoRA dropout must be between 0 and 1, got {args.lora_dropout}")
        
        if not args.lora_target_modules:
            raise ValueError("LoRA target modules cannot be empty when use_lora is True")

def _parse_data_type(data_type_str: str) -> torch.dtype:

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    
    if data_type_str not in dtype_map:
        raise ValueError(f"Not supported data type: {data_type_str}")
    
    return dtype_map[data_type_str]

def config_register(addition_config: Optional[Dict[str, Any]] = None) -> Tuple[ModelConfig, TrainingConfig]:

    try:
        parser = parse_args()
        
        # 构建 LoRA 配置
        lora_config: Optional[LoraConfig] = None
        if parser.use_lora:
            # 安全处理 lora_target_modules
            target_modules = []
            if parser.lora_target_modules:
                target_modules = parser.lora_target_modules.split(",")
            
            if parser.lora_path != "":
                lora_config = LoraConfig({
                    'lora_path': parser.lora_path,
                })
            else:
                lora_config = LoraConfig({
                    "lora_path": parser.lora_path,
                    "lora_rank": parser.lora_rank,
                    "lora_alpha": parser.lora_alpha,
                    "lora_target_modules": target_modules,
                    "lora_dropout": parser.lora_dropout,
                })
            
            logger.info(f"LoRA target modules: {target_modules}")

        model_config = ModelConfig({
            "model_path_to_load": parser.model_path_to_load,
            "use_lora": parser.use_lora,
            "lora_config": lora_config,
            "data_type": _parse_data_type(parser.data_type), 
            "system_prompt": parser.system_prompt,
            "use_deepspeed": parser.use_deepspeed,
            "deepspeed_cfg_path": parser.deepspeed_cfg_path,
            "gradit_checkpoing": getattr(parser, "enable_gradit_checkpoing", False)

        })
        
        training_config = TrainingConfig({
            "epoch": parser.epoch,
            "lr": parser.lr,
            "batch_size": parser.batch_size,
            "gradient_accumulation_steps": parser.gradient_accumulation_steps,
            "data_path": parser.data_path,
            "output_dir": parser.output_dir,
            "seed": parser.seed,
            "use_tensorboard": parser.use_tensorboard,
            "tensorboard_path": parser.tensorboard_path if parser.tensorboard_path != '' else parser.output_dir,
            "use_process_bar": parser.use_process_bar,
            "training_type": parser.training_type,
            "checkpoint_epoch": [int(ep) for ep in parser.checkpoint_epoch.split(',') if ep != ''],
            "local_rank": parser.local_rank,
            "world_size": 1,
        })

        # Add additional configs by TrainingType
        training_type = TrainingType(parser.training_type)


        if training_type == TrainingType.SFT.value:
            training_config.update({
                "use_NLIRG": getattr(parser, "use_NLIRG", False), 
                "token_batch": getattr(parser, "token_batch", -1),
                "Distillition": getattr(parser, "Distillition", False),
                "teacher_model_path": getattr(parser, "teacher_model_path", ""),
                "coefficient_of_origin_loss": getattr(parser, "coefficient_of_origin_loss", 0.0),
                "max_seq_len": getattr(parser, "max_seq_len", 20520),
                
            })
            logger.info(f"SFT training mode selected with NLIRG={getattr(parser, 'use_NLIRG', False)}")
        elif training_type == TrainingType.CPT.value:
            training_config.update({
                "pack_length": getattr(parser, "pack_length", -1),
                "use_NLIRG": getattr(parser, "use_NLIRG", False),
            })
            logger.info(f"CPT training mode selected with pack_length={getattr(parser, 'pack_length', -1)}")
        elif training_type == TrainingType.RL.value:
            training_config.update({
                "use_NLIRG": getattr(parser, "use_NLIRG", False),
                "token_batch": getattr(parser, "token_batch", -1),
                "Distillition": getattr(parser, "Distillition", False),
                "teacher_model_path": getattr(parser, "teacher_model_path", ""),
                "coefficient_of_origin_loss": getattr(parser, "coefficient_of_origin_loss", 0.0),
                "max_seq_len": getattr(parser, "max_seq_len", 20520),
                "rl_type": getattr(parser, "rl_type", "agent"),
            })
            logger.info("RL training mode selected")
        else:
            logger.warning(f"Unknown training type: {training_type}, using default configuration")


        logger.info(f"Model path: {parser.model_path_to_load}")
        logger.info(f"Data path: {parser.data_path}")
        logger.info(f"Output directory: {parser.output_dir}")
        
        # save config to output_dir
        try:
            if parser.local_rank != 0:
                return model_config, training_config
            config_dir = os.path.join(parser.output_dir, "configs")
            os.makedirs(config_dir, exist_ok=True)
            
            # save model_config and training_config to config_dir
            model_config_path = os.path.join(config_dir, "model_config.json")
            training_config_path = os.path.join(config_dir, "training_config.json")
            model_config_dict = dict(model_config.to_dict())

            model_config_dict["data_type"] = parser.data_type
            model_config_to_save = ModelConfig(model_config_dict)
            model_config_to_save.save(model_config_path)

            training_config_to_save = dict(training_config.to_dict())
            training_config_to_save['checkpoint_epoch'] = ','.join(str(ep) for ep in training_config_to_save['checkpoint_epoch'])

            training_config.save(training_config_path)
            
            # save combined config to config_dir
            combined_config_path = os.path.join(config_dir, "config.json")
            save_combined_config(model_config_to_save, training_config, combined_config_path)
            del model_config_dict, model_config_to_save

            
            logger.info(f"Config saved to: {config_dir}")
        except Exception as e:
            logger.warning(f"Failed to save config: {str(e)}")
        
        return model_config, training_config
        
    except Exception as e:
        logger.error(f"Error in config registration: {str(e)}")
        raise

def save_combined_config(model_config: ModelConfig, training_config: TrainingConfig, file_path: str) -> None:
    """
    Merges the model and training configs and saves them to a JSON file.
    
    Args:
        model_config
        training_config 
        file_path
    """

    from zoneinfo import ZoneInfo
    tz = ZoneInfo("Asia/Shanghai")
    # create a combined config dictionary
    combined_config = {
        "model_config": model_config.to_dict(),
        "training_config": training_config.to_dict(),
        "start_time": datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %z"),
        "version": "1.0"
    }
    
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(combined_config, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved combined config to: {file_path}")


def load_combined_config(file_path: str) -> Tuple[ModelConfig, TrainingConfig]:
    """
    Load the combined config from a JSON file.
    
    Args:
        file_path: JSON path
        
    Returns:
        Tuple[ModelConfig, TrainingConfig]: A tuple containing the model and training configs.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        combined_config = json.load(f)
    model_config_dict = combined_config.get("model_config", {})
    model_config_dict['data_type'] = _parse_data_type(model_config_dict.get("data_type", ""))
    lora_config = LoraConfig(model_config_dict.get("lora_config", {}))
    model_config_dict["lora_config"] = lora_config
    # if model_config_dict["use_lora"]:
    #     model_config_dict["lora_config"][""]

    model_config = ModelConfig(model_config_dict)
    training_config = TrainingConfig(combined_config.get("training_config", {}))
    
    logger.info(f"Config loaded from {file_path}.")
    return model_config, training_config


def validate_config(model_config: ModelConfig, training_config: TrainingConfig) -> bool:
    """
    Validate the model and training configs.
    
    Args:
        model_config
        training_config
        
    Returns:
        bool: 
    """
    if not os.path.exists(model_config["model_path"]):
        logger.warning(f"Model path does not exist: {model_config['model_path']}")
    
    if not os.path.exists(model_config["data_path"]):
        logger.warning(f"Data path does not exist: {model_config['data_path']}")
    
    if model_config["use_lora"] and not model_config["lora_config"]:
        raise ValueError("LoRA is enabled but lora_config is None")
    
    if training_config["epoch"] <= 0:
        raise ValueError(f"Epoch must be positive, got {training_config['epoch']}")
    
    if training_config["lr"] <= 0:
        raise ValueError(f"Learning rate must be positive, got {training_config['lr']}")
    
    training_type = training_config["training_type"]
    if training_type == TrainingType.SFT.value:
        if "use_NLIRG" not in training_config:
            logger.warning("use_NLIRG not found in training_config for SFT training")
    elif training_type == TrainingType.CPT.value:
        if "pack_length" not in training_config:
            logger.warning("pack_length not found in training_config for CPT training")
    
    logger.info("Configuration validation passed")
    return True