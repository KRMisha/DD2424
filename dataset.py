import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode

TEST_DATASET_SIZE = 200


class KvasirSegDataset(Dataset):
    def __init__(self, root, train, transform=None):
        self.root = root
        self.transform = transform

        self.images = sorted((root / 'images').glob('*.jpg'))
        self.masks = sorted((root / 'masks').glob('*.jpg'))

        if train:
            self.images = self.images[:-TEST_DATASET_SIZE]
            self.masks = self.masks[:-TEST_DATASET_SIZE]
        else:
            self.images = self.images[-TEST_DATASET_SIZE:]
            self.masks = self.masks[-TEST_DATASET_SIZE:]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = read_image(str(self.images[idx])).type(torch.float) / 255
        mask = read_image(str(self.masks[idx]), ImageReadMode.GRAY).type(torch.float) / 255
        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask
