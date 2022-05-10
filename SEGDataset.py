import os
import numpy as np
import torch
from PIL import Image


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
        img = Image.open(img_path).convert("RGB").resize((img_size,img_size))
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path).resize((img_size,img_size))

        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        mask = np.array(mask>50,dtype=int) #Make the white part = 1 and the black part = 0

        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = np.array(mask == obj_ids[:, None, None], dtype=int)
        masks = 255*masks
        # masks = torch.as_tensor(masks, dtype=torch.uint8)

        if self.transforms is not None:
            img= self.transforms(img)
            masks= self.transforms(masks)
        return img, masks

    def __len__(self):
        return len(self.imgs)

