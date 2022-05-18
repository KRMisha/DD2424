#!/usr/bin/env python3

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm import trange
import config
from dataset import KvasirSegDataset
import functions # TODO: Reconsider
from model import UNet
from predict import make_predictions
from train import train, valid


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
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Train model
    train_losses = []
    valid_losses = []
    with trange(config.EPOCHS, desc='Training network') as t:
        for _ in t:
            train_loss = train(train_dataloader, model, loss_fn, optimizer)
            train_losses.append(train_loss)

            valid_loss = valid(valid_dataloader, model, loss_fn)
            valid_losses.append(valid_loss)

            t.set_postfix(train_loss=train_loss, valid_loss=valid_loss)
    print('Network training complete')

    # Plot training curve
    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(range(1, len(train_losses) + 1), train_losses, label='Training')
    ax.plot(range(1, len(valid_losses) + 1), valid_losses, label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Average loss for each epoch')
    ax.legend()
    fig.savefig(str(config.TRAINING_PLOT_PATH), bbox_inches='tight')
    plt.close(fig)

    # Save trained model
    torch.save(model, config.MODEL_PATH)
    print('Model saved to disk')

    # Load testing dataset and create testing data loader
    test_dataset = KvasirSegDataset(root=config.DATASET_DIRECTORY, train=False, transform=functions.transforms)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=0, # TODO: num_workers=os.cpu_count()?
        pin_memory=config.PIN_MEMORY
    )

    # Load trained model
    model = torch.load(config.MODEL_PATH).to(config.DEVICE)
    print('Model loaded from disk')

    # Test model
    # TODO: Improve
    # TODO: Could use CLI arg to predict only, or a separate script
    make_predictions(model, test_dataloader)


if __name__ == '__main__':
    main()
