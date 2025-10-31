from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM
from torch.nn import functional as F
from training_code.utils.common_utils import dynamic_sigmoid_batch, masked_mean
from training_code.core.model.model_saver import save_checkpoint_model
import logging
from training_code.core.log.processing_log import ProcessLogger

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def per_epoch_train(model, optimizer, batcher, training_config, epoch, teacher_model=None, process_logger:ProcessLogger=None):

    loss_threshold = 3.0
    loss_deadline = 15.0


    device = f'cuda:{training_config.local_rank}'

    is_distilltion = training_config.Distillition

    batch_loop = tqdm(batcher, desc=f"Epoch {epoch}", leave=True)

    for step, batch in enumerate(batch_loop, 1):
        try:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            input_ids = batch.pop('input_ids')
            attention_mask = batch.pop('attention_mask')

            if is_distilltion:

                with torch.no_grad():
                    input_ids_teacher = input_ids.clone().detach().to('cuda:1')
                    attention_mask_teacher = attention_mask.clone().detach().to('cuda:1')
                    teacher_output = teacher_model(
                        input_ids=input_ids_teacher,
                        attention_mask=attention_mask_teacher,
                    )
                    teacher_logits = teacher_output.logits.view(-1,teacher_output.logits.shape[-1]).to(device)



            labels = batch.pop('labels')
            token_batch = training_config.token_batch
            

            
            loss_logs = []
            label_copy = labels.clone().detach()
            to_train_token_idx = torch.where(label_copy.view(-1) != -100)[0]
            if to_train_token_idx.numel() > 0:
                to_train_token_idx_start = to_train_token_idx[0]

                num_token_epoch_to_train = (labels.shape[-1]-to_train_token_idx_start)//token_batch + 1
                for token_ep in range(num_token_epoch_to_train):
                    # (start position, end position)
                    label_range = (to_train_token_idx_start + token_ep*token_batch, to_train_token_idx_start + token_ep*token_batch + token_batch)
                    
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    logits_flatten = outputs.logits.view(-1,outputs.logits.shape[-1])[label_range[0]:label_range[1]]

                    targets = labels.clone().detach().view(-1)[label_range[0]:label_range[1]]

                    # calculate unweighted loss
                    loss_unweighted = torch.nn.functional.cross_entropy(
                        logits_flatten, targets, reduction='none'
                    )

                    log_loss = loss_unweighted.clone().detach()

                    if is_distilltion:
                        
                        prediction_softmax = F.log_softmax(logits_flatten, dim=-1)
                        target_source_softmax = F.softmax(teacher_logits[label_range[0]:label_range[1]], dim=-1)
                        origin_loss_coefficient = training_config.coefficient_of_origin_loss
                        
                        backward_loss = (1 - origin_loss_coefficient) * -1 * (target_source_softmax * prediction_softmax).sum(dim=-1) + (origin_loss_coefficient * loss_unweighted)
                        
                    else:
                        backward_loss = loss_unweighted

                    if training_config.use_NLIRG:
                        mask = (log_loss > loss_threshold) & (log_loss < loss_deadline)
                        with torch.no_grad():

                            weights = dynamic_sigmoid_batch(
                                log_loss, 
                                max_lr=1,
                                loss_threshold=loss_threshold, 
                                loss_deadline=loss_deadline
                            )
                        
                        # calculate weighted loss
                        loss_weighted_unmeand = backward_loss * weights
                    
                        masked_loss_weighted = masked_mean(loss_weighted_unmeand)
                        masked_loss_log = masked_mean(log_loss)
                        loss_log = masked_loss_log.item()
                    
                    else:
                        
                        mask = log_loss != 0
                        loss_weighted_unmeand = backward_loss

                        masked_loss_weighted = loss_weighted_unmeand.mean()
                        masked_loss_log = log_loss.mean()
                        loss_log = masked_loss_log.item()
                        if loss_log == None: log_loss = 0

                    loss_logs.append(loss_log)

                    loss_weighted_unmeand = loss_weighted_unmeand[mask]


                    if loss_weighted_unmeand.shape[-1] != 0:
                        if optimizer != None:

                            masked_loss_weighted.backward()
                            optimizer.step()
                            optimizer.zero_grad()
                        
                        else:
                            model.backward(masked_loss_weighted)
                            model.step()
                process_logger.write_tb_log(epoch, step, sum(loss_logs)/len(loss_logs))
                del loss_log, logits_flatten, loss_unweighted, targets, outputs, backward_loss, mask
                if 'teacher_output' in locals():
                    del teacher_output
                if 'teacher_logits' in locals():
                    del teacher_logits
                mean_loss = sum(loss_logs)/len(loss_logs)
            else:
                process_logger.write_tb_log(epoch, step, 0)
                mean_loss = 0
                
        
            batch_loop.set_postfix({
                'progress': f"{batch_loop.n}/{batch_loop.total}",
                'avg_loss': f"{mean_loss:.4f}",
            })


        except Exception as e:
            logger.error(f"\nError in step {step}: {e}")
            raise

    return model



def sft_train_model(model, optimizer, batcher, training_config, model_config, tensorboard_writer=None):
    model.train()
    
    if training_config.Distillition:
        teacher_model = AutoModelForCausalLM.from_pretrained(training_config.teacher_model_path, torch_dtype=torch.bfloat16)

        teacher_model.to(model.device)
        teacher_model.eval()
    else:
        teacher_model = None

    # unified training loop
    for epoch in range(0, training_config.epoch):
        
        model = per_epoch_train(model, optimizer, batcher, training_config, epoch, teacher_model, tensorboard_writer)
        save_checkpoint_model(model, training_config, model_config, epoch)

    return model
        
