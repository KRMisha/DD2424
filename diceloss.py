import torch
from torch import nn

class DiceLoss(nn.Module):
    """Create a custome loss to compute Dice Loss"""
    def __init__(self, epsilon=1) -> None:
        super(DiceLoss, self).__init__()
        self.epsilon=1 #epsilon avoid us to divide by 0. It's also on the top to still allow us to have a loss of 0 if we find everything corretly.
    def forward(self, input, target):
        sig=nn.Sigmoid()
        input=sig(input)
        num = 2 * torch.sum(input * target) + self.epsilon
        denom = torch.sum(input) + torch.sum(target) + self.epsilon
        return 1 - num/denom