#!/usr/bin/env python3

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import config
from dataset import KvasirSegDataset
import functions # TODO: Reconsider
from model import UNet
from train import train


def main():
    # Load dataset and split it into train and test sets
    dataset = KvasirSegDataset(root=config.DATASET_PATH, transform=functions.transforms)
    train_size = int(len(dataset) * (1 - config.TEST_SPLIT))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size]) # TODO: Use validation set?

    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0, # TODO: num_workers=os.cpu_count()?
        pin_memory=config.PIN_MEMORY
    )
    test_dataloader = DataLoader(
        test_dataset,
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
    train(train_dataloader, test_dataloader, model, loss_function, optimizer)

    # TODO: Predict (could use CLI arg to predict only, or a separate script)


if __name__ == '__main__':
    main()
