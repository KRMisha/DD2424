import config
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T

# Inspired by https://amaarora.github.io/2020/09/13/unet.html


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_relu_stack = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv_relu_stack(x)


class Encoder(nn.Module):
    def __init__(self, channels=(3, 16, 32, 64)):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(in_channels, out_channels) for in_channels, out_channels in zip(channels, channels[1:])
        ])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        # Store intermediate outputs
        block_outputs = []

        for block in self.blocks:
            # Apply encoder block
            x = block(x)
            block_outputs.append(x)

            # Apply max pooling
            x = self.pool(x)

        return block_outputs


class Decoder(nn.Module):
    def __init__(self, channels=(64, 32, 16)):
        super().__init__()
        self.up_convolutions = nn.ModuleList([
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2) for in_channels, out_channels in zip(channels, channels[1:])
        ])
        self.blocks = nn.ModuleList([
            Block(in_channels, out_channels) for in_channels, out_channels in zip(channels, channels[1:])
        ])

    def forward(self, x, encoder_block_outputs):
        for up_convolution, block, encoder_block_output in zip(self.up_convolutions, self.blocks, encoder_block_outputs):
            # Perform up-convolution
            x = up_convolution(x)

            # Crop intermediate features from encoder path and concatenate them with upsampled features
            cropped_size = x.shape[-2:]
            cropped_encoder_block_output = T.CenterCrop(cropped_size)(encoder_block_output)
            x = torch.cat([x, cropped_encoder_block_output], dim=1)

            # Apply decoder block
            x = block(x)

        return x


class UNet(nn.Module):
    def __init__(self, encChannels=(3, 16, 32, 64),
                 decChannels=(64, 32, 16),
                 nbClasses=1, retainDim=True,
                 outSize=(config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)):
        super().__init__()
        # initialize the encoder and decoder
        self.encoder = Encoder(encChannels)
        self.decoder = Decoder(decChannels)
        # initialize the regression head and store the class variables
        self.head = nn.Conv2d(decChannels[-1], nbClasses, 1)
        self.retainDim = retainDim
        self.outSize = outSize

    def forward(self, x):
        # grab the features from the encoder
        encFeatures = self.encoder(x)
        # pass the encoder features through decoder making sure that
        # their dimensions are suited for concatenation
        decFeatures = self.decoder(encFeatures[::-1][0],
                                   encFeatures[::-1][1:])
        # pass the decoder features through the regression head to
        # obtain the segmentation mask
        map = self.head(decFeatures)
        # check to see if we are retaining the original output
        # dimensions and if so, then resize the output to match them
        if self.retainDim:
            map = F.interpolate(map, self.outSize)
        # return the segmentation map
        return map
