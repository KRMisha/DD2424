import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def prepare_plot(origImage, origMask, predMask,id):
    # initialize our figure
    figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
    # plot the original image, its mask, and the predicted mask
    ax[0].imshow(origImage.permute(1, 2, 0))
    ax[1].imshow(origMask[0], cmap='gray')
    ax[2].imshow(predMask[0], cmap='gray')
    # set the titles of the subplots
    ax[0].set_title("Image")
    ax[1].set_title("Original Mask")
    ax[2].set_title("Predicted Mask")
    # set the layout of the figure and display it
    figure.tight_layout()
    plt.savefig(os.path.join(config.TEST_IMAGES_PATHS,str(id)+'test_save.jpg'))
    plt.close(figure)


def make_predictions(model, test_dataloader):
    # set model to evaluation mode
    model.eval()
    # turn off gradient tracking
    acc_list=[]
    with torch.no_grad():
        # load the image from disk, swap its color channels, cast it
        # to float data type, and scale its pixel values
        for i,(images, gtMasks) in enumerate(test_dataloader):
            # Getting the predicted mask thanks to the image and the model
            predMasks = model(images)
            predMasks = torch.sigmoid(predMasks)
            predMasks = predMasks
            # filter out the weak predictions and convert them to integers
            predMasks = (predMasks > config.THRESHOLD) * 255
            # prepare a plot for visualization
            for j in range(len(images)):
                prepare_plot(images[j], gtMasks[j], predMasks[j], i*config.BATCH_SIZE+j)
                # Compute the accuracy
                acc_list.append(1-abs(gtMasks[j]-predMasks[j]).sum()/np.prod(images[j].shape[1:3]))
    acc=np.mean(np.array(acc_list))
    print("accuracy : ", acc)
