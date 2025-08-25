import torch
import torch.nn as nn
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, targets, 
                      temperature=2.0, distillation_weight=0.7, criterion=nn.BCEWithLogitsLoss()):
    """
    Compute distillation loss for binary classification (sigmoid-based).
    
    Args:
        student_logits (Tensor): Raw logits from student model
        teacher_logits (Tensor): Raw logits from teacher model
        targets (Tensor): Ground truth labels
        temperature (float): Temperature for softening probabilities
        distillation_weight (float): Weight of the distillation loss
        criterion: Loss function for original task (default: BCEWithLogitsLoss)

    Returns:
        loss (Tensor): Final combined loss
    """

    soft_targets =nn.functional.sigmoid(teacher_logits / temperature)
    soft_prob =nn.functional.sigmoid(student_logits / temperature)


    soft_targets_loss = torch.sum(
            soft_targets * (soft_targets * (1 - soft_prob) / (soft_prob * (1 - soft_targets))).log() +
            ((1 - soft_targets) / (1 - soft_prob)).log()
        ) / (soft_prob.size()[0] * soft_prob.size()[1]) * (temperature ** 2)
    
    label_loss = criterion(student_logits, targets)

  
    loss = distillation_weight * soft_targets_loss + (1 - distillation_weight) * label_loss

    return loss
