from training_code.core.config.globle_config import config_register
from training_code.core.data_processer.main_processer import data_processer
from training_code.core.model.model_loader import model_processer, load_tokenizer, initialize_distributed
from training_code.core.trainer.main_trainer import training_processer
from training_code.core.model.model_saver import save_model
from training_code.core.log.processing_log import ProcessLogger
import os
WORLD_SIZE = 1
LOCAL_RANK = 0
os.environ["TOKENIZERS_PARALLELISM"] = 'true'
def main():
    
    model_config, training_config = config_register()
    
    process_logger = ProcessLogger(training_config, model_config)

    initialize_distributed(model_config,training_config)

    tokenizer = load_tokenizer(model_config)

    data_batcher = data_processer(tokenizer, training_config, model_config)
    
    model, batcher, optimizer = model_processer(data_batcher, tokenizer, model_config, training_config)

    trained_model = training_processer(model, optimizer, batcher, tokenizer, training_config, model_config, process_logger)

    save_model(trained_model, tokenizer, model_config, training_config)


if __name__ == "__main__":
    main()