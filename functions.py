from torchvision import transforms
import config

transforms = transforms.Compose([
    transforms.Resize(config.INPUT_IMAGE_DIMENSIONS),
])
