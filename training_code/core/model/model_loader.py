from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys
from training_code.core.config.globle_config import ModelConfig, TrainingConfig
import torch
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model
import deepspeed
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader 
from functools import partial
from training_code.core.data_processer.sft_processer import collate_fn_sft
from training_code.core.data_processer.cpt_processer import collate_fn_cpt
from training_code.core.config.globle_config import TrainingType
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def initialize_distributed(model_config, training_config):
    """For initializing distributed training environment."""
    
    if model_config.use_deepspeed:
        global WORLD_SIZE, LOCAL_RANK
        import deepspeed
        deepspeed.init_distributed()
        LOCAL_RANK = int(os.environ.get('LOCAL_RANK',-1))
        WORLD_SIZE = dist.get_world_size()
        training_config.word_size = WORLD_SIZE
        
        torch.cuda.set_device(LOCAL_RANK)
    
        if LOCAL_RANK == 0:
            logger.info(f"Initializing distributed training environment: local rank {LOCAL_RANK} with {WORLD_SIZE} gpus.")

def create_ds_config(batcher, training_config):
    global WORLD_SIZE
    
    steps_per_epoch = (len(batcher) + WORLD_SIZE - 1) // WORLD_SIZE

    total_training_steps = steps_per_epoch * training_config.epoch
    warmup_steps = int(total_training_steps * 0.01)  # 10% for warmup
    
    
    # create config dict
    ds_config = {
        "gather_16bit_weights_on_model_save": True,
        "train_micro_batch_size_per_gpu": training_config.batch_size,
        "gradient_accumulation_steps": training_config.gradient_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": training_config.lr,
                "weight_decay": 0.01
            }
        },
        "bf16": {"enabled": True},
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": training_config.lr,
                "total_num_steps": total_training_steps,
                "warmup_num_steps": warmup_steps
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



def configure_lora(model, model_config):

    if model_config.lora_config.lora_path == '':
        model.enable_input_require_grads()
        peft_config = LoraConfig(
            r=model_config.lora_config.lora_rank,
            lora_alpha=model_config.lora_config.lora_alpha,
            target_modules=model_config.lora_config.lora_target_modules,
            lora_dropout=model_config.lora_config.lora_dropout,
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)


    else:
        if not os.path.exists(model_config.lora_config.lora_path):
            raise ValueError(f"LoRA path does not exist: {model_config.lora_config.lora_path}")
        
        model = PeftModel.from_pretrained(model, model_config.lora_config.lora_path, is_trainable=True)

        for name, m in model.named_modules():
            if 'lora' in name:
                m.requires_grad_(True)
            if hasattr(m, "gradient_checkpointing") and not m.gradient_checkpointing:
                m.gradient_checkpointing = True

    model.print_trainable_parameters()

    # print(model)

    return model

def load_tokenizer(model_cfg: ModelConfig):
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_path_to_load, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def load_model(model_cfg: ModelConfig):
    data_type = model_cfg.data_type
    if not torch.cuda.is_available():
        data_type = torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_cfg.model_path_to_load, trust_remote_code=True, torch_dtype=data_type)
    if model_cfg.gradit_checkpoing:
        model.gradient_checkpointing_enable()
    if model_cfg.use_lora:
        model = configure_lora(model, model_cfg)

    return model



def model_processer(batcher, tokenizer, model_cfg: ModelConfig, training_cfg: TrainingConfig):
    model = load_model(model_cfg)
    model.train()

    optimizer = None

    if training_cfg.training_type == TrainingType.SFT:
        collate_fn = partial(collate_fn_sft, tokenizer=tokenizer)
    elif training_cfg.training_type == TrainingType.CPT:
        collate_fn = collate_fn_cpt
    else:
        raise ValueError(f"Unsupported training type: {training_cfg.training_type}.")

    if model_cfg.use_deepspeed:
        if model_cfg.deepspeed_cfg_path != "":
            import json
            with open(model_cfg.deepspeed_cfg_path, "r") as f:
                ds_config_dict = json.load(f)
        else:
            ds_config_dict = create_ds_config(batcher, training_cfg)

        sampler = DistributedSampler(batcher, rank=training_cfg.local_rank, num_replicas=dist.get_world_size(), shuffle=False)
        batcher = DataLoader(batcher, batch_size=training_cfg.batch_size, shuffle=False, collate_fn=collate_fn, sampler=sampler)
        model, _, _, _ = deepspeed.initialize(model=model, config=ds_config_dict)

    else:
        device = f'cuda:{training_cfg.local_rank}'
        model.to(device)
        batcher = DataLoader(batcher, batch_size=training_cfg.batch_size, shuffle=False, collate_fn=collate_fn)
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=training_cfg.lr)

    return model, batcher, optimizer