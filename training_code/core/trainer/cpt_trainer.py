import torch
from training_code.core.log.processing_log import ProcessLogger
from training_code.utils.common_utils import dynamic_sigmoid_batch
import logging
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def compute_weighted_loss(model_engine, input_ids, attention_mask, labels, traing_config):
    """Compute weighted loss."""

    outputs = model_engine(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        return_dict=True
    )
    logits = outputs.logits
    original_loss = outputs.loss

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    if shift_logits.numel() == 0 or shift_labels.numel() == 0:
        return original_loss, original_loss, []
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
    loss_per_token_original = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )

    if traing_config.use_NLIRG:
        with torch.no_grad():
            weight = dynamic_sigmoid_batch(loss_per_token_original.detach(), max_lr=1, x0=1.2, min_lr=5e-8, k=1.7, loss_threshold=3.0, loss_deadline=15.0)
        loss_per_token = loss_per_token_original * weight
    else:
        loss_per_token = loss_per_token_original

    loss_per_token_original = loss_per_token_original.detach().view(shift_labels.shape)
    loss_per_token = loss_per_token.view(shift_labels.shape)

      
    valid_mask = (shift_labels != -100)
    
    if not traing_config.use_NLIRG:
        return original_loss, original_loss

    valid_mask = (shift_labels != -100)
    valid_loss = loss_per_token[valid_mask]

    if valid_loss.numel() == 0:
        return original_loss, original_loss

    weighted_loss = valid_loss.mean()
    return weighted_loss, original_loss


def prepare_4d_attention_mask(attention_mask_with_indices, dtype=torch.bfloat16):
    """Generate 4D attention mask from 2D attention mask with indices."""
    bsz, seq_len = attention_mask_with_indices.size()
    min_dtype = torch.finfo(dtype).min
    
    expanded_mask = attention_mask_with_indices[:, None, None, :].expand(bsz, 1, seq_len, seq_len)
    padding_mask = torch.where(expanded_mask != 0, 1, 0).to(dtype=dtype)
    segment_mask = torch.eq(
        expanded_mask, 
        expanded_mask.transpose(-1, -2)
    ).int()
    causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.int, device=attention_mask_with_indices.device))
    
    combined_mask = segment_mask * causal_mask * padding_mask
    attention_mask_4d = torch.where(
        combined_mask != 0, 
        torch.tensor(0, dtype=dtype, device=attention_mask_with_indices.device), 
        min_dtype
    )
    
    return attention_mask_4d

def cpt_train_model(model, optimizer, cpt_data_loader, traing_config, process_logger:ProcessLogger):
    """main pretrain"""
    
    for epoch in range(traing_config.epoch):
        model.train()
        if traing_config.local_rank == 0:
            logger.info(f"Start training Epoch {epoch+1}/{traing_config.epoch}.")
        
        batch_loop = tqdm(cpt_data_loader, desc=f"Epoch {epoch+1}", leave=True)       
        
        for step, batch_cpt in enumerate(batch_loop,1):
            device = f'cuda:{traing_config.local_rank}'
            input_ids = batch_cpt["input_ids"].to(device)
            segment_ids = batch_cpt["segment_ids"].to(device)
            sample_id = batch_cpt.get("id", f"cpt_{step}")
            attention_mask_4d = prepare_4d_attention_mask(segment_ids, dtype=torch.bfloat16)
            
            weighted_loss, origin_loss = compute_weighted_loss(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask_4d,
                labels=input_ids,
                traing_config=traing_config
            )
            mean_loss = origin_loss.item()

            process_logger.write_tb_log(epoch, step, mean_loss)

            if traing_config.local_rank == 0:
                batch_loop.set_postfix({
                    'progress': f"{batch_loop.n}/{batch_loop.total}",
                    'loss':f"{mean_loss:.4f}", 
                })

            if optimizer == None:
                model.backward(weighted_loss)
                model.step()
            else:
                optimizer.zero_grad()
                weighted_loss.backward()
                optimizer.step()
    
                

    return model

        
