import torch


def dice_coefficient(input, target):
    intersection = torch.sum(input * target)
    return 2 * intersection / (torch.sum(input) + torch.sum(target))


def iou(input, target):
    intersection = torch.sum(input * target)
    union = torch.sum(input) + torch.sum(target) - intersection
    return intersection / union
