import matplotlib.pyplot as plt
import torch
import config
from diceloss import DiceLoss


def plot_segmentation(x, y, pred, i):
    fig, axs = plt.subplots(1, 3, constrained_layout=True)

    axs[0].imshow(x.cpu().permute(1, 2, 0))
    axs[1].imshow(y.cpu().squeeze(), cmap='gray')
    axs[2].imshow(pred.cpu().squeeze(), cmap='gray')

    axs[0].axis('off')
    axs[1].axis('off')
    axs[2].axis('off')

    axs[0].set_title('Image')
    axs[1].set_title('Ground truth mask')
    axs[2].set_title('Predicted mask')

    fig.savefig(str(config.PREDICTED_IMAGES_DIRECTORY / f'{i}.jpg'), bbox_inches='tight')
    plt.close(fig)


def test(dataloader, model):
    model.eval()
    loss_fn=DiceLoss()
    dice_coeff = 0

    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)

            pred = model(x)
            pred_to_acc=pred
            pred = (torch.sigmoid(pred) > config.THRESHOLD).float()

            plot_segmentation(x[0], y[0], pred[0], i)

            # Compute the dice_coeff
            dice_coeff += 1 - loss_fn(pred_to_acc, y)

    pixel_accuracy = dice_coeff / len(dataloader)
    print(f'Dice coefficient: {pixel_accuracy}')
