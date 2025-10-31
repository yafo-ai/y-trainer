from training_code.core.config.globle_config import TrainingType
from .cpt_processer import process_cpt_data
from .sft_processer import process_sft_data

    
def data_processer(tokenizer, training_config, model_config):
    
    if training_config.training_type == TrainingType.CPT:
        return process_cpt_data(tokenizer, training_config=training_config)
    elif training_config.training_type == TrainingType.SFT:
        return process_sft_data(tokenizer, training_config=training_config, model_config=model_config)
    else:
        raise ValueError(f"Unsupported training type: {training_config.training_type}")
