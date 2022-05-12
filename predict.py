# USAGE
# python predict.py
# import the necessary packages
import config
import functions
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import json
from SEGDataset import SEGDataset
from torch.utils.data import DataLoader


def prepare_plot(origImage, origMask, predMask):
    # initialize our figure
    figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
    # plot the original image, its mask, and the predicted mask
    ax[0].imshow(origImage.permute(1, 2, 0))
    ax[1].imshow(origMask[0], cmap='gray')
    ax[2].imshow(predMask)
    # set the titles of the subplots
    ax[0].set_title("Image")
    ax[1].set_title("Original Mask")
    ax[2].set_title("Predicted Mask")
    # set the layout of the figure and display it
    figure.tight_layout()
    plt.savefig(os.path.join(config.BASE_OUTPUT,'test_save.jpg'))


def make_predictions(model, imageNames):
    # set model to evaluation mode
    model.eval()
    # turn off gradient tracking
    with torch.no_grad():
        testDS = SEGDataset(root=config.DATASET_PATH, name_list=imageNames, transforms=functions.transforms)
        testLoader = DataLoader(testDS, shuffle=True, batch_size=2, pin_memory=config.PIN_MEMORY,
                                 num_workers=0)
        # load the image from disk, swap its color channels, cast it
        # to float data type, and scale its pixel values
        for images, gtMasks in testLoader:
            # Getting the predicted mask thanks to the image and the model
            predMasks = model(images).squeeze()
            predMasks = torch.sigmoid(predMasks)
            predMasks = predMasks
            # filter out the weak predictions and convert them to integers
            predMasks = (predMasks > config.THRESHOLD) * 255
            # prepare a plot for visualization
            for i in range(len(images)):
                prepare_plot(images[i], gtMasks[i], predMasks[i])

# load the image paths in our testing file and randomly select 10
# image paths
name_list = open(config.TEST_PATHS).read().strip().split("\n")
name_list = np.random.choice(name_list, size=10)

# load our model from disk and flash it to the current device
print("[INFO] load up model...")
unet = torch.load(config.MODEL_PATH).to(config.DEVICE)

# make predictions and visualize the results
make_predictions(unet, name_list)


