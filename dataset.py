import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
import config


class KvasirSegDataset(Dataset):
    def __init__(self, root, train, transform=None):
        self.root = root
        self.transform = transform

        self.images = sorted((root / 'images').glob('*.jpg'))
        self.masks = sorted((root / 'masks').glob('*.jpg'))

        if config.TRAIN_DATASET_SIZE + config.VALID_DATASET_SIZE + config.TEST_DATASET_SIZE > len(self.images):
            raise ValueError('The total size of the training, validation, and test datasets exceeds the size of the full dataset.')

        if train:
            total_dataset_size = config.TRAIN_DATASET_SIZE + config.VALID_DATASET_SIZE
            self.images = self.images[:total_dataset_size]
            self.masks = self.masks[:total_dataset_size]
        else:
            self.images = self.images[-config.TEST_DATASET_SIZE:]
            self.masks = self.masks[-config.TEST_DATASET_SIZE:]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = read_image(str(self.images[idx])).type(torch.float) / 255
        mask = read_image(str(self.masks[idx]), ImageReadMode.GRAY).type(torch.float) / 255
        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask
