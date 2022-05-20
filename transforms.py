from torchvision import transforms
import config

base_transforms = transforms.Compose([
    transforms.Resize(config.INPUT_IMAGE_DIMENSIONS),
])

invert = transforms.Compose([
    transforms.Resize(config.INPUT_IMAGE_DIMENSIONS),
    transforms.RandomInvert(p=0.5)
])

rotate = transforms.Compose([
    transforms.Resize(config.INPUT_IMAGE_DIMENSIONS),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5)
])

crop = transforms.Compose([
    transforms.Resize(config.INPUT_IMAGE_DIMENSIONS),
    transforms.RandomCrop(size=(64,64)),
    transforms.Resize(config.INPUT_IMAGE_DIMENSIONS),
])

random = transforms.Compose([
    transforms.Resize(config.INPUT_IMAGE_DIMENSIONS),
    transforms.RandAugment(), # to be specified
    transforms.Resize(config.INPUT_IMAGE_DIMENSIONS)
])

color = transforms.Compose([
    transforms.Resize(config.INPUT_IMAGE_DIMENSIONS),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
])
