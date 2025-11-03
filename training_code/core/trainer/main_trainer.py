from .cpt_trainer import cpt_train_model
from .sft_trainer import sft_train_model
from training_code.core.config.globle_config import TrainingType

def training_processer(model, optimizer, batcher, tokenizer, training_config, model_config=None, process_logger=None):
    if training_config.training_type == TrainingType.CPT:
        return cpt_train_model(model, optimizer, batcher, training_config, process_logger)
    elif training_config.training_type == TrainingType.SFT:
        return sft_train_model(model, tokenizer, optimizer, batcher, training_config, model_config, process_logger)
    else:
        raise ValueError(f"Not supported training type: {training_config.training_type}")
