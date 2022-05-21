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
PIN_MEMORY = DEVICE == 'cuda'

# Training, validation, and test dataset sizes
TRAIN_DATASET_SIZE = 800
VALID_DATASET_SIZE = 100
TEST_DATASET_SIZE = 100

# Training hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 16
EPOCHS = 50

# Neural network parameters
# ENCODER_CHANNELS = (3, 64, 128, 256, 512, 1024)
# DECODER_CHANNELS = (1024, 512, 256, 128, 64)
ENCODER_CHANNELS = (3, 16, 32, 64, 128)
DECODER_CHANNELS = (128, 64, 32, 16)

# Threshold to convert sigmoid to binary class
THRESHOLD = 0.5

# Resized input image dimensions
INPUT_IMAGE_DIMENSIONS = (512, 512)
