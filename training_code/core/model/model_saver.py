import os
from contextlib import nullcontext
import torch.distributed as dist
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
def save_model(model, tokenizer, model_config, training_config):

    if training_config.local_rank == 0:
        print("Saving model...")
    
    tokenizer.save_pretrained(training_config.output_dir)
    print(f"Now saving model to {training_config.output_dir}...")

    try:
        if getattr(model_config,'use_deepspeed', False):
            # 条件导入deepspeed
            from deepspeed import zero
            gather_context = (
                zero.GatheredParameters(list(model.module.parameters()), modifier_rank=0)
                if model.zero_optimization_stage() == 3
                else nullcontext()
            )
            with gather_context:
                if training_config.local_rank == 0:
                    os.makedirs(training_config.output_dir, exist_ok=True)
                    model.module.save_pretrained(training_config.output_dir)
        else:
            model.save_pretrained(training_config.output_dir)
        logger.info(f"✅ Model saved to: {training_config.output_dir}")

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
    
    if training_config.local_rank == 0:
        print(f"✅ Finished training.")


def save_checkpoint_model(model, tokenizer, model_config, training_config, epoch):

    Checkpoint_epoch = training_config.checkpoint_epoch if isinstance(training_config.checkpoint_epoch, list) else [training_config.checkpoint_epoch]

    if Checkpoint_epoch == []:
        return model
    
    if epoch in [e for e in Checkpoint_epoch]:
        if model_config.use_deepspeed:
            if training_config.local_rank == 0:
                logger.warning('Deepespeed checkpoint saving is not supported yet.')
            return model
        current_checkpoint_dir = f"{training_config.output_dir}/checkpoint-{epoch+1}"
        model.save_pretrained(
            save_directory=current_checkpoint_dir,
            save_adapter=model_config.use_lora,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(current_checkpoint_dir)
        logger.info(f"\n=== saved epoch {epoch}，module saved to {current_checkpoint_dir} ===")
        
    return model
