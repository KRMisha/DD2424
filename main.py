#!/usr/bin/env python3

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import config
from dataset import KvasirSegDataset
import functions # TODO: Reconsider
from model import UNet
from predict import make_predictions
from train import train


def main():
    # Ensure output directories exist
    config.OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    config.PREDICTED_IMAGES_DIRECTORY.mkdir(exist_ok=True)

    print(f'Using {config.DEVICE} device')

    # Load training dataset and split it into train and validation sets
    dataset = KvasirSegDataset(root=config.DATASET_DIRECTORY, train=True, transform=functions.transforms)
    train_dataset, valid_dataset = random_split(dataset, [config.TRAIN_DATASET_SIZE, config.VALID_DATASET_SIZE])
    print(
        f'Training: {config.TRAIN_DATASET_SIZE} samples | '
        f'Validation: {config.VALID_DATASET_SIZE} samples | '
        f'Test: {config.TEST_DATASET_SIZE} samples'
    )

    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0, # TODO: num_workers=os.cpu_count()?
        pin_memory=config.PIN_MEMORY
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0, # TODO: num_workers=os.cpu_count()?
        pin_memory=config.PIN_MEMORY
    )

    # Initialize model
    model = UNet(encChannels=config.ENCCHANNELS, decChannels=config.DECCHANNELS).to(config.DEVICE)

    # Loss function and optimizer
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.INIT_LR)

    # Train model
    # TODO: Improve (see official PyTorch tutorial conventions)
    train(train_dataloader, valid_dataloader, model, loss_function, optimizer)

    # Load testing dataset and create testing data loader
    test_dataset = KvasirSegDataset(root=config.DATASET_DIRECTORY, train=False, transform=functions.transforms)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=0, # TODO: num_workers=os.cpu_count()?
        pin_memory=config.PIN_MEMORY
    )

    # Test model
    # TODO: Improve
    # TODO: Could use CLI arg to predict only, or a separate script
    model = torch.load(config.MODEL_PATH).to(config.DEVICE)
    make_predictions(model, test_dataloader)


if __name__ == '__main__':
    main()
