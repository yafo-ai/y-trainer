import random
import numpy as np
import torch
import math

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU


def masked_mean(losses: torch.Tensor) -> torch.Tensor:
    """
    calculate average non-zero loss，token loss=0 is not included.
    
    :param losses: 
    :return:
    """
    mask = (losses != 0)
    
    # select all loss that is not 0
    valid_losses = losses[mask]
    
    num_valid = valid_losses.numel()

    if num_valid > 0:
        return valid_losses.mean()
    else:
        return torch.tensor(0.0, device=losses.device)


def _calculate_sigmoid_weights(exponent, max_lr, min_lr=0):
    """
    calculate lr
    
    :param exponent: 
    :param max_lr: 
    :param min_lr: 
    :return: 
    """
    # safity edge
    if exponent > 88:
        return max_lr + min_lr
    elif exponent < -88:
        return min_lr
    exp_term = math.exp(exponent)
    return max_lr / (1 + exp_term) + 1e-7 + min_lr

def dynamic_sigmoid(loss, max_lr=1, x0=1.2, k=1.7, min_lr=5e-8, loss_threshold=3.0, loss_deadline=15.0) -> float:
    """
    
    :param loss: current loss
    :param max_lr: 
    :param x0: 
    :param min_lr: 
    :param k: 
    :param loss_threshold: 
    :param loss_deadline: 

    """
    if loss <= loss_threshold:
        # （0.0 <= loss < loss_threshold）
        exponent = -k * (loss - x0)
        return _calculate_sigmoid_weights(exponent, max_lr)

    elif loss_threshold < loss < loss_deadline:
        # （loss_threshold <= loss < loss_deadline）
        exponent = 1*loss - 6.2
        return _calculate_sigmoid_weights(exponent, max_lr, min_lr)
    else:
        # loss >= loss_deadline）
        return 0.0



def dynamic_sigmoid_batch(losses: torch.Tensor,
                          max_lr: float = 1.0,
                          x0: float = 1.2,
                          min_lr: float = 5e-8,
                          k: float = 1.7,
                          loss_threshold: float = 3.0,
                          loss_deadline: float = 15.0) -> torch.Tensor:
    """

    Dynamically Parameterized Sigmoid Function (Batch Version)

    Args:

    losses: A batch of loss values, with shape [batch_size], type torch.Tensor

    max_lr: Maximum learning rate, the peak of the curve

    x0: Midpoint, where the learning rate is halved

    min_lr: Minimum learning rate, the lowest point of the curve

    k: Curve slope, the larger the value, the shorter the distance between the highest and lowest points

    loss_threshold: Boundary point separating easy and hard learning intervals, using different sigmoid learning rates for these two intervals

    loss_deadline: Loss cutoff point where the learning rate drops to zero; after exceeding this value, the learning rate is set to 0

    return: Dynamic learning rate corresponding to each loss, with the same shape as losses [batch_size]

    """
    learning_rates = torch.zeros_like(losses)

    # 1：loss <= loss_threshold
    mask1 = losses <= loss_threshold
    if mask1.any():  
        loss_masked = losses[mask1]
        exponent = -k * (loss_masked - x0)
        
        safe_exponent = torch.clamp(exponent, min=-709, max=709)
        exp_term = torch.exp(safe_exponent)
        sigmoid_lr = max_lr / (1 + exp_term) + torch.finfo(torch.float32).eps
        
        learning_rates[mask1] = sigmoid_lr

    # 2：loss_threshold < loss < loss_deadline
    mask2 = (losses > loss_threshold) & (losses < loss_deadline)
    if mask2.any():
        loss_masked = losses[mask2]
        exponent = 1*loss_masked - 6.2 
        safe_exponent = torch.clamp(exponent, min=-709, max=709)
        exp_term = torch.exp(safe_exponent)
        sigmoid_part = max_lr / (1 + exp_term) + torch.finfo(torch.float32).eps
        sigmoid_lr = sigmoid_part + min_lr
        learning_rates[mask2] = sigmoid_lr

    # 3：loss >= loss_deadline
    mask3 = losses >= loss_deadline
    if mask3.any():
        learning_rates[mask3] = 0.0

    return learning_rates


if __name__ == '__main__':
    print(dynamic_sigmoid(0.000446))