import torchvision.transforms as T
import config

base = T.Compose([
    T.Resize(config.INPUT_IMAGE_DIMENSIONS),
])

augmentations = T.Compose([
    T.RandomVerticalFlip(p=0.5),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomApply([T.RandomAffine(degrees=(0, 45), translate=(0, 0.2), scale=(0.8, 1))], p=0.3),
    T.RandomApply([T.RandomAffine(degrees=0, shear=(0, 30))], p=0.7),
    T.RandomApply([T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)], p=0.5),
])
