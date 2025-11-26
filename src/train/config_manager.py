from argparse import ArgumentParser, Namespace
import logging
from src.train.config import TrainConfig,TrainingType, parse_data_type

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class TrainConfigManager:

    @staticmethod
    def _parse_args() -> Namespace:

        parser = ArgumentParser(description="Training configuration parser")
        
        # model config
        parser.add_argument("--model_path_to_load", type=str, required=True, help="Path to the model")

        parser.add_argument("--data_type", type=str, default="bfloat16", 
                            help="Data type for training, float16 or float32", 
                            choices=["float16", "float32", "bfloat16"])
        
        parser.add_argument("--system_prompt", type=str, default="You are a helpful assistant.", help="System prompt")

        # LoRA config
        parser.add_argument("--use_lora", type=str, default="false", help="Whether to use LoRA")
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

        parser.add_argument('--save_optimizer_state', type=str, default="false", help='whether to save optimizer state')
        parser.add_argument('--use_deepspeed', type=str, default="false", help='whether to use deepspeed config')
        parser.add_argument('--deepspeed_cfg_path', type=str, default="", help='deepspeed config path, if not set, will automaticaly generate one with stage 3')
        parser.add_argument('--local_rank', type=int, default=0, help="Rank for current process")
        parser.add_argument('--enable_gradit_checkpoing', type=str, default="false", help='whether to set gradit checkpoing')

        # cpt config
        parser.add_argument("--pack_length", type=int, default=-1, 
                            help="Pack length for training, -1 means no pack")

        # sft config
        parser.add_argument("--use_NLIRG", type=str, default="false", help="Whether to use NLIRG")
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
        parser.add_argument("--use_tensorboard", type=str, default="false", help="Whether to use tensorboard")
        parser.add_argument("--tensorboard_path", type=str, default="", 
                            help="User can specify the path to tensorboard, default is the output_dir")
        
        parser.add_argument("--use_process_bar", type=str, default="false", help="Whether to use process bar")

        args = parser.parse_args()

        return args
    
    @staticmethod
    def register_from_inputs()->TrainConfig:

        try:
            parser = TrainConfigManager._parse_args()
            
            inputs={}
            
            if parser.use_lora == 'true':
                # 安全处理 lora_target_modules
                target_modules = []
                if parser.lora_target_modules:
                    target_modules = parser.lora_target_modules.split(",")
                
                if parser.lora_path != "":
                    inputs.update({
                        'lora_path': parser.lora_path,
                    })
                    logger.info(f"LoRA in {parser.lora_path} is loading")
                else:
                    inputs.update({
                        "lora_path": parser.lora_path,
                        "lora_rank": parser.lora_rank,
                        "lora_alpha": parser.lora_alpha,
                        "lora_target_modules": target_modules,
                        "lora_dropout": parser.lora_dropout,
                    })
                    logger.info(f"LoRA target modules: {target_modules}")

            
            inputs.update({
                "model_path_to_load": parser.model_path_to_load,
                "use_lora": parser.use_lora == 'true',
                "data_type": parse_data_type(parser.data_type), 
                "system_prompt": parser.system_prompt,
                "use_deepspeed": parser.use_deepspeed=='true',
                "deepspeed_cfg_path": parser.deepspeed_cfg_path,
                "gradit_checkpoing": getattr(parser, "enable_gradit_checkpoing", 'false') == 'true',
                "epoch": parser.epoch,
                "lr": parser.lr,
                "batch_size": parser.batch_size,
                "gradient_accumulation_steps": parser.gradient_accumulation_steps,
                "data_path": parser.data_path,
                "output_dir": parser.output_dir,
                "seed": parser.seed,
                "use_tensorboard": parser.use_tensorboard=='true',
                "tensorboard_path": parser.tensorboard_path if parser.tensorboard_path != '' else parser.output_dir,
                "use_process_bar": parser.use_process_bar=='true',
                "training_type": parser.training_type,
                "checkpoint_epoch": [int(ep) for ep in parser.checkpoint_epoch.split(',') if ep != ''],
                "local_rank": parser.local_rank,
                "world_size": 1,
            })

            training_type = TrainingType(parser.training_type)


            if training_type == TrainingType.SFT.value:
                inputs.update({
                    "use_nlirg": getattr(parser, "use_NLIRG", 'false')=='true', 
                    "token_batch": getattr(parser, "token_batch", -1),
                    "distillition": getattr(parser, "Distillition", False),
                    "teacher_model_path": getattr(parser, "teacher_model_path", ""),
                    "coefficient_of_origin_loss": getattr(parser, "coefficient_of_origin_loss", 0.0),
                    "max_seq_len": getattr(parser, "max_seq_len", 20520),
                    
                })
                logger.info(f"SFT training mode selected with NLIRG={getattr(parser, 'use_NLIRG', 'false')=='true'}")
            elif training_type == TrainingType.CPT.value:
                inputs.update({
                    "pack_length": getattr(parser, "pack_length", -1),
                    "use_nlirg": getattr(parser, "use_NLIRG", 'false')=='true',
                })
                logger.info(f"CPT training mode selected with pack_length={getattr(parser, 'pack_length', -1)}")
            else:
                logger.warning(f"Unknown training type: {training_type}, using default configuration")


            logger.info(f"Model path: {parser.model_path_to_load}")
            logger.info(f"Data path: {parser.data_path}")
            logger.info(f"Output directory: {parser.output_dir}")
            
            train_args=TrainConfig(**inputs)

            return train_args
            
        except Exception as e:
            logger.error(f"Error in config registration: {str(e)}")
            raise

    @staticmethod
    def register_from_dict(args_dict: dict) -> TrainConfig:
        """
        Register training arguments from a dictionary.
        """
        try:
            if isinstance(args_dict.get("training_type","sft"), str):
                args_dict["training_type"] = TrainingType(args_dict["training_type"])
            
            if isinstance(args_dict.get("data_type"),str):
                args_dict["data_type"] = parse_data_type(args_dict["data_type"])
                
            train_args = TrainConfig(**args_dict)
            return train_args
        except Exception as e:
            logger.error(f"Error in config registration from dict: {str(e)}")
            raise



if __name__ == "__main__":
    args_dict = {
        "model_path_to_load": "E:\GPT\LLM_models\Qwen2.5-0.5B-Instruct",
        "use_lora": True,
        "system_prompt": "You are a helpful assistant.",
        "use_deepspeed": False,
        "deepspeed_cfg_path": "",
        "gradit_checkpoing": False,
        "epoch": 3,
        "lr": 1e-5,
        "batch_size": 2,
        "gradient_accumulation_steps": 1,
        "data_path": ".\example_dataset",
        "output_dir": "E:\a",
        "seed": 42,
        "use_tensorboard": False,
        "tensorboard_path": "",
        "use_process_bar": True,
        "training_type": TrainingType.SFT.value,
        "checkpoint_epoch": [],
        "local_rank": 0,
        "world_size": 1,
    }

    TrainConfigManager.register_from_dict(args_dict)