import torch
from torch import nn


class DiceLoss(nn.Module):
    """Custom loss function to compute DICE loss."""

    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = torch.sigmoid(input)
        numerator = 2 * torch.sum(input * target) + torch.finfo(torch.float32).eps
        denominator = torch.sum(input) + torch.sum(target) + torch.finfo(torch.float32).eps
        return 1 - numerator / denominator
