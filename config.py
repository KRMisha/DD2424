from pathlib import Path
import torch

# Dataset path
DATASET_DIRECTORY = Path('data')

# Output paths
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
BATCH_SIZE = 40
EPOCHS = 200

# Neural network parameters
ENCODER_CHANNELS = (3, 128, 256, 512)
DECODER_CHANNELS = (512, 256, 128)

# Threshold to convert sigmoid to binary class
THRESHOLD = 0.5

# Resized input image dimensions
INPUT_IMAGE_DIMENSIONS = (140, 140)
