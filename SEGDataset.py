import os
import numpy as np
import torch
from PIL import Image, ImageOps


# Found there : #https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

#Constants
img_size=512


class SEGDataset(torch.utils.data.Dataset):
    def __init__(self, root, name_list, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned

        self.imgs=list(map(lambda name : name+'.jpg', name_list))
        self.masks = list(map(lambda name: name + '.jpg', name_list))



    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        mask_path = os.path.join(self.root, "masks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        masks = Image.open(mask_path).convert("1")
        if self.transforms is not None:
            img= self.transforms(img)
            masks= self.transforms(masks)
        return img, masks
    def __len__(self):
        return len(self.imgs)

