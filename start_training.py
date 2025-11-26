import logging
import os
from src.train.config_manager import TrainConfigManager
from src.train.train_engine import TrainEngine
WORLD_SIZE = 1
LOCAL_RANK = 0
os.environ["TOKENIZERS_PARALLELISM"] = 'true'



def test():
    inputs={
        "model_path_to_load": "/raid1/big-models/Qwen2.5-0.5B-Instruct/",
        "training_type": "sft",
        "epoch": 3,
        "use_nlirg": True,
        "checkpoint_epoch": [0,1],
        "data_path": "./example_dataset/v72_sft_all_0909.json",
        "output_dir": "./LLM_models/TEST_SFT",
        "batch_size": 1,
        "token_batch": 10,
        "lora_target_modules": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        "use_tensorboard":True,
        "tensorboard_path":"/raid1/tensorboard_log/"
    }
    config=TrainConfigManager.register_from_dict(inputs)
    train_engine=TrainEngine(config)
    train_engine.start_train()


def main():

    config=TrainConfigManager.register_from_inputs()
    train_engine=TrainEngine(config)
    train_engine.start_train()

if __name__ == "__main__":
    main()