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

# Device settings for training
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PIN_MEMORY = DEVICE == 'cuda' # TODO: Investigate this parameter

# Training, validation, and test dataset sizes
TRAIN_DATASET_SIZE = 760
VALID_DATASET_SIZE = 120
TEST_DATASET_SIZE = 120

# Training hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 16
EPOCHS = 50

# Neural network parameters
# ENCCHANNELS = (3, 64, 128, 256, 512, 1024)
# DECCHANNELS = (1024, 512, 256, 128, 64)
ENCCHANNELS = (3, 16, 32, 64, 128)
DECCHANNELS = (128, 64, 32, 16)

# Threshold to filter out weak predictions
THRESHOLD = 0.5

# Resized input image dimensions
INPUT_IMAGE_WIDTH = 512
INPUT_IMAGE_HEIGHT = 512
