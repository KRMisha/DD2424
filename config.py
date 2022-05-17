from pathlib import Path
import torch
import os

# Dataset path
DATASET_PATH = Path('data')

# define the test split
TEST_SPLIT = 0.20
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

# define the path to the base output directory
BASE_OUTPUT = "output"
# define the path to the output serialized model, model training
# plot, and testing image paths
MODEL_PATH = os.path.join(BASE_OUTPUT, "unet_tgs_salt.pth")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot-deep"+ str(NUM_LEVELS) + "-Starting : " + str(ENCCHANNELS[1])+".png"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])
TEST_IMAGES_PATHS = os.path.sep.join([BASE_OUTPUT, "images_output"])
