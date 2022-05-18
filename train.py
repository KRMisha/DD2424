import config
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time

# TODO: Improve (split this function into train and validation during training)
def train(train_dataloader, valid_dataloader, model, loss_function, optimizer):

    # calculate steps per epoch for training and validation set
    trainSteps = len(train_dataloader)
    validSteps = len(valid_dataloader)
    # initialize a dictionary to store training history
    H = {"train_loss": [], "valid_loss": []}

    # loop over epochs
    print("[INFO] training the network...")
    startTime = time.time()
    for e in tqdm(range(config.NUM_EPOCHS)):
        # set the model in training mode
        model.train()
        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalValidLoss = 0
        # loop over the training set
        for (i, (x, y)) in enumerate(train_dataloader):
            # send the input to the device
            (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
            # perform a forward pass and calculate the training loss
            pred = model(x)
            loss = loss_function(pred, y)
            # first, zero out any previously accumulated gradients, then
            # perform backpropagation, and then update model parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # add the loss to the total training loss so far
            totalTrainLoss += loss
        # switch off autograd
        with torch.no_grad():
            # set the model in evaluation mode

            model.eval()
            # loop over the validation set
            for (x, y) in valid_dataloader:
                # send the input to the device
                (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
                # make the predictions and calculate the validation loss
                pred = model(x)
                totalValidLoss += loss_function(pred, y)
        # calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValidLoss = totalValidLoss / validSteps
        # update our training history
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["valid_loss"].append(avgValidLoss.cpu().detach().numpy())
        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
        print("Train loss: {:.6f}, Validation loss: {:.4f}".format(
            avgTrainLoss, avgValidLoss))
    # display the total time needed to perform the training
    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(
        endTime - startTime))

    # plot the training loss
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["valid_loss"], label="valid_loss")
    plt.title("Training Loss while training UNet - Deep : "+ str(config.NUM_LEVELS) + "- Starting : " + str(config.ENCCHANNELS[1]))
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(config.PLOT_PATH)

    # serialize the model to disk
    torch.save(model, config.MODEL_PATH)
