from torchvision import transforms
import config

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((config.INPUT_IMAGE_HEIGHT,config.INPUT_IMAGE_WIDTH))])