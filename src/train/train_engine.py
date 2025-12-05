from contextlib import nullcontext
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys
import torch
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader 
from functools import partial
from src.train.config import TrainConfig,TrainingType
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



class TrainEngine:

    def __init__(self, config:TrainConfig):
        self._config = config
        self.tokenizer=None
        self.model=None
        self.batcher=None
        self.optimizer=None
        self.trainer=None

    def _trainer_builder(self):
        """
        加载训练器
        """
        if self._config.training_type == TrainingType.SFT:
            from src.train.trainer_sft import TrainerSFT
            self.trainer = TrainerSFT(self._config,self.save_checkpoint_model)
        elif self._config.training_type == TrainingType.CPT:
            from src.train.trainer_cpt import TrainerCPT
            self.trainer = TrainerCPT(self._config)
        else:
            raise ValueError(f"Unknown training type: {self._config.training_type}")
        
    def _setup_training(self):
        """
        初始化训练环境
        """
        self._init_distributed()
        self._trainer_builder()
        tokenizer=self._load_tokenizer()
        self.tokenizer=tokenizer
        dataset=self.trainer.load_dataset(self.tokenizer)
        model, batcher, optimizer=self._model_processer(dataset, tokenizer)
        self.model=model
        self.batcher=batcher
        self.optimizer=optimizer

    def start_train(self):
        """
        开始训练
        """
        try:
            self._setup_training()
            techer=self._load_teacher_model(self.model)
            self.trainer.train(self.model,self.optimizer,self.batcher,techer)
            self._save_model()
        except Exception as e:
            logger.error(f"❌ train failed: {e}")
            print(f"❌ train failed: {e}")
        finally:
            del self.model
            self.tokenizer=None
            self.model=None
            self.optimizer=None
            self.trainer=None
            self.batcher=None
            gc.collect()
             
    def _init_distributed(self):
        """初始化分布式训练环境."""
        
        if self._config.use_deepspeed:
            global WORLD_SIZE, LOCAL_RANK
            
            try:
                import deepspeed
            except ImportError:
                logger.error("DeepSpeed is not installed. Please install deepspeed.")
                raise

            deepspeed.init_distributed()
            LOCAL_RANK = int(os.environ.get('LOCAL_RANK',-1))
            WORLD_SIZE = dist.get_world_size()
            self._config.world_size = WORLD_SIZE
            
            torch.cuda.set_device(LOCAL_RANK)

            if self._config.world_size>10 and self._config.use_nlirg:
                logger.error("NLIRG is not supported in distributed training with more than 10 gpus.")
                raise ValueError("NLIRG is not supported in distributed training with more than 10 gpus.")
                
        
            if LOCAL_RANK == 0:
                logger.info(f"Initializing distributed training environment: local rank {LOCAL_RANK} with {WORLD_SIZE} gpus.")

    def _load_tokenizer(self):
        """
        加载tokenizer
        """
        tokenizer = AutoTokenizer.from_pretrained(self._config.model_path_to_load, trust_remote_code=True)
        tokenizer.padding_side = 'left'
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token


        return tokenizer
    
    def _load_model(self):
        """
        加载模型
        """
        data_type = self._config.data_type
        if not torch.cuda.is_available():
            data_type = torch.float32
        model = AutoModelForCausalLM.from_pretrained(self._config.model_path_to_load, trust_remote_code=True, torch_dtype=data_type)
        if self._config.gradit_checkpoing:
            model.gradient_checkpointing_enable()
        if self._config.use_lora:
            model = self._configure_lora(model)

        model.train()

        return model
    
    def _load_teacher_model(self,model):

        if self._config.distillition:

            teacher_model = AutoModelForCausalLM.from_pretrained(self._config.teacher_model_path, torch_dtype=torch.bfloat16)

            teacher_model.to(model.device)
            teacher_model.eval()

    def _configure_lora(self, model):

        if self._config.lora_path == '':
            model.enable_input_require_grads()
            peft_config = LoraConfig(
                r=self._config.lora_rank,
                lora_alpha=self._config.lora_alpha,
                target_modules=self._config.lora_target_modules,
                lora_dropout=self._config.lora_dropout,
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, peft_config)

        else:
            if not os.path.exists(self._config.lora_path):
                raise ValueError(f"LoRA path does not exist: {self._config.lora_path}")
            
            model = PeftModel.from_pretrained(model, self._config.lora_path, is_trainable=True)

            for name, m in model.named_modules():
                if 'lora' in name:
                    m.requires_grad_(True)

        model.print_trainable_parameters()

        return model

    def _distributed_config(self,dataset_size):
        global WORLD_SIZE
        
        steps_per_epoch = (dataset_size + WORLD_SIZE - 1) // WORLD_SIZE
        total_num_steps = steps_per_epoch * self._config.epoch
        warmup_num_steps = int(total_num_steps * 0.01)  # 10% for warmup
        
        
        # create config dict
        ds_config = {
            "gather_16bit_weights_on_model_save": True,
            "train_micro_batch_size_per_gpu": self._config.batch_size,
            "gradient_accumulation_steps": self._config.gradient_accumulation_steps,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": self._config.lr,
                    "weight_decay": 0.01
                }
            },
            "bf16": {"enabled": True},
            "scheduler": {
                "type": "WarmupDecayLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": self._config.lr,
                    "total_num_steps": total_num_steps,
                    "warmup_num_steps": warmup_num_steps
                }
            },
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {"device": "none"},
                "offload_param": {"device": "none"},
                "overlap_comm": True,
                "contiguous_gradients": True
            },
            "activation_checkpointing": {
                "partition_activations": True,
                "contiguous_memory_optimization": True,
                "number_checkpoints": 1
            },
            "steps_per_print": 100,
        }

        
        return ds_config

    def _model_processer(self,dataset, tokenizer):
        model = self._load_model()
        optimizer = None
        collate_fn = partial(self.trainer.collate_fn, tokenizer=tokenizer)

        if self._config.use_deepspeed:
            import deepspeed
            if self._config.deepspeed_cfg_path != "":
                import json
                with open(self._config.deepspeed_cfg_path, "r") as f:
                    ds_config_dict = json.load(f)
            else:
                dataset_size=len(dataset)
                ds_config_dict = self._distributed_config(dataset_size)

            sampler = DistributedSampler(dataset, rank=self._config.local_rank, num_replicas=dist.get_world_size(), shuffle=False)
            batcher = DataLoader(dataset, batch_size=self._config.batch_size, shuffle=False, collate_fn=collate_fn, sampler=sampler)
            model, _, _, _ = deepspeed.initialize(model=model, config=ds_config_dict)

        else:
            device = f'cuda:{self._config.local_rank}'
            model.to(device)
            batcher = DataLoader(dataset, batch_size=self._config.batch_size, shuffle=False, collate_fn=collate_fn)
            trainable_params = filter(lambda p: p.requires_grad, model.parameters())
            optimizer = torch.optim.AdamW(trainable_params, lr=self._config.lr)

        return model, batcher, optimizer
    
    def _save_model(self):

        if self._config.local_rank == 0:
            print("Saving model...")
        
        self.tokenizer.save_pretrained(self._config.output_dir)
        print(f"Now saving model to {self._config.output_dir}...")

        try:
            if self._config.use_deepspeed:
                # 条件导入deepspeed
                from deepspeed import zero
                gather_context = (
                    zero.GatheredParameters(list(self.model.module.parameters()), modifier_rank=0)
                    if self.model.zero_optimization_stage() == 3
                    else nullcontext()
                )
                with gather_context:
                    if self._config.local_rank == 0:
                        os.makedirs(self._config.output_dir, exist_ok=True)
                        self.model.module.save_pretrained(self._config.output_dir)
            else:
                self.model.save_pretrained(self._config.output_dir)
            logger.info(f"✅ Model saved to: {self._config.output_dir}")

        except Exception as save_error:
            print(f"Error in model saving: {save_error}")
            import traceback
            traceback.print_exc()
        
        if dist.is_initialized():
            try:
                dist.barrier()
            except:
                pass
            try:
                dist.destroy_process_group()
            except:
                pass
        
        if self._config.local_rank == 0:
            print(f"✅ Finished training.")

    def save_checkpoint_model(self,epoch):

        checkpoint_epoch = self._config.checkpoint_epoch if isinstance(self._config.checkpoint_epoch, list) else [self._config.checkpoint_epoch]

        if checkpoint_epoch == []:
            return
        
        if epoch in [e for e in checkpoint_epoch]:
            if self._config.use_deepspeed:
                if self._config.local_rank == 0:
                    logger.warning('Deepespeed checkpoint saving is not supported yet.')
                return
            current_checkpoint_dir = f"{self._config.output_dir}/checkpoint-{epoch+1}"
            self.model.save_pretrained(
                save_directory=current_checkpoint_dir,
                save_adapter=self._config.use_lora,
                safe_serialization=True,
            )
            self.tokenizer.save_pretrained(current_checkpoint_dir)
            logger.info(f"\n=== saved epoch {epoch}，module saved to {current_checkpoint_dir} ===")



def merge_lora_with_base(base_model_path, lora_path, output_path):
    # 加载基础模型 (BF16精度)
    print(f"Loading base model from: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    
    # 加载LoRA适配器并合并
    print(f"Loading LoRA adapter from: {lora_path}")
    merged_model = PeftModel.from_pretrained(base_model, lora_path)
    merged_model = merged_model.merge_and_unload()
    
    # 保存合并后的模型
    print(f"Saving merged model to: {output_path}")
    merged_model.save_pretrained(output_path)
    
    # 同时保存tokenizer (使用基础模型的tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)
    print("Merge completed successfully!")
            