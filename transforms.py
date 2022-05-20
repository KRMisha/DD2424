from torchvision import transforms
import config

# TODO: Combine these transforms together to have a single composition for data augmentations

base = transforms.Compose([
    transforms.Resize(config.INPUT_IMAGE_DIMENSIONS),
])

invert = transforms.Compose([
    base,
    transforms.RandomInvert(p=0.5),
])

rotate = transforms.Compose([
    base,
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
])

crop = transforms.Compose([
    base,
    transforms.RandomCrop(size=(64, 64)),
    transforms.Resize(config.INPUT_IMAGE_DIMENSIONS),
])

random = transforms.Compose([
    base,
    transforms.RandAugment(), # TODO: Specify
    transforms.Resize(config.INPUT_IMAGE_DIMENSIONS),
])

color = transforms.Compose([
    base,
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
])
