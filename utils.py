import torch
import torch.nn.functional as F


def average_displacement_error(y, y_pred):
    mask = (y != -1).to(y.device)
    return torch.sum((F.mse_loss(y_pred, y, reduction='none') * mask).sum(dim=2).sqrt()) / (torch.sum(mask)/2)
