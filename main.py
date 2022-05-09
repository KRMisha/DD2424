#!/usr/bin/env python3
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import json
from SEGDataset import SEGDataset


def get_transform():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize(255),
                                    transforms.CenterCrop(224)])
    return transform

def main():

    data= SEGDataset('data',transforms.ToTensor())
    # print(data.__len__())
    print(data.__getitem__(2))

if __name__ == '__main__':
    main()




