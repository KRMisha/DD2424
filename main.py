#!/usr/bin/env python3
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from SEGDataset import SEGDataset
import json
from NeuralNetwork import UNet

def show_image_masks(img, masks):
    fig, axs = plt.subplots(2)
    axs[0].imshow(img.permute(1, 2, 0))
    axs[1].imshow(((-1 * masks) + 1) * 255)
    plt.show()

def train_test_split(prop_of_train=0.9):
    with open('data/kavsir_bboxes.json', 'r') as json_file:
        json_dict = json.load(json_file)
    name_list = list(json_dict.keys())
    number_of_train=int(prop_of_train*len(name_list))
    train_names, test_names = torch.utils.data.random_split(name_list,[number_of_train, len(name_list)-number_of_train])
    return train_names, test_names

def main():
    # Showing images
    train_names, test_names = train_test_split(0.8)
    train_data= SEGDataset('data',train_names, transforms.ToTensor())
    n_data=0
    img, masks=train_data.__getitem__(n_data)
    # show_image_masks(img, masks)

    #Creating a Network
    x = torch.randn(size=(1, 1, 512, 512), dtype=torch.float32)
    print(x.shape)
    print(img.shape)
    Unet_network=UNet(in_channels=1)
    print(Unet_network.forward(x).shape)

if __name__ == '__main__':
    main()




