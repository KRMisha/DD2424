import matplotlib.pyplot as plt
import numpy as np
import torch
import config


def plot_segmentation(x, y, pred, i):
    fig, axs = plt.subplots(1, 3, constrained_layout=True)

    axs[0].imshow(x.permute(1, 2, 0))
    axs[1].imshow(y.squeeze(), cmap='gray')
    axs[2].imshow(pred.squeeze(), cmap='gray')

    axs[0].axis('off')
    axs[1].axis('off')
    axs[2].axis('off')

    axs[0].set_title('Image')
    axs[1].set_title('Ground truth mask')
    axs[2].set_title('Predicted mask')

    fig.savefig(str(config.PREDICTED_IMAGES_DIRECTORY / f'{i}.jpg'), bbox_inches='tight')
    plt.close(fig)


def predict(dataloader, model):
    model.eval()

    total_pixel_accuracy = 0

    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)

            pred = model(x)
            pred = (torch.sigmoid(pred) > config.THRESHOLD).float()

            plot_segmentation(x[0], y[0], pred[0], i)

            # Compute pixel accuracy
            # TODO: Replace/extend this with DICE, IOU or other metric more commonly used for segmentation
            total_pixel_accuracy += 1 - np.mean(np.abs(y[0].squeeze() - pred[0].squeeze()).numpy())

    pixel_accuracy = total_pixel_accuracy / len(dataloader)
    print(f'Pixel accuracy on test data: {pixel_accuracy}')
