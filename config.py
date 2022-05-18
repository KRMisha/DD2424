from pathlib import Path
import torch

# Dataset path
DATASET_DIRECTORY = Path('data')

# Output paths
# TODO: Make these names specific to each experiment when later running multiple experiments
OUTPUT_DIRECTORY = Path('output')
MODEL_PATH = OUTPUT_DIRECTORY / 'model.pth'
TRAINING_PLOT_PATH = OUTPUT_DIRECTORY / 'training.png'
PREDICTED_IMAGES_DIRECTORY = OUTPUT_DIRECTORY / 'predictions'

# define the test split
TRAIN_VALID_SPLIT_RATIO = 0.20
# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

#Define the depth of the network:
# ENCCHANNELS=(3, 64, 128, 256, 512, 1024)
# DECCHANNELS=(1024, 512, 256, 128, 64)
ENCCHANNELS=(3, 16, 32, 64, 128)
DECCHANNELS=(128, 64, 32, 16)

# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
NUM_CHANNELS = 3
NUM_CLASSES = 1
NUM_LEVELS = len(DECCHANNELS)
# initialize learning rate, number of epochs to train for, and the
# batch size
NB_IMAGES=1000
INIT_LR = 0.001
NUM_EPOCHS = 50
# BATCH_SIZE = 64
BATCH_SIZE = 16
# define the input image dimensions
INPUT_IMAGE_WIDTH = 512
INPUT_IMAGE_HEIGHT = 512
# define threshold to filter weak predictions
THRESHOLD = 0.5
