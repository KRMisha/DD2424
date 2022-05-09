#!/usr/bin/env python3
import torch
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from SEGDataset import SEGDataset

def show_image_masks(img, target):
    fig, axs = plt.subplots(2)
    axs[0].imshow(img.permute(1, 2, 0))
    axs[1].imshow(((-1 * target['masks']) + 1) * 255)
    plt.show()

def main():
    data= SEGDataset('data',transforms.ToTensor())
    n_data=0
    img, target=data.__getitem__(n_data)
    show_image_masks(img, target)


if __name__ == '__main__':
    main()




