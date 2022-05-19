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
    def __init__(self, channels):
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
    def __init__(self, channels):
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
    def __init__(
        self,
        encoder_channels=(3, 16, 32, 64),
        decoder_channels=(64, 32, 16),
        output_classes=1,
        retain_dimensions=True
    ):
        super().__init__()
        self.encoder = Encoder(encoder_channels)
        self.decoder = Decoder(decoder_channels)
        self.segmentation_map = nn.Conv2d(decoder_channels[-1], output_classes, 1)
        self.retain_dimensions = retain_dimensions

    def forward(self, x):
        # Contracting path
        encoder_block_outputs = self.encoder(x)

        # Expansive path
        output = self.decoder(encoder_block_outputs[-1], reversed(encoder_block_outputs[:-1]))

        # Output segmentation map
        output = self.segmentation_map(output)

        # Resize output to input image dimensions if enabled
        if self.retain_dimensions:
            output = F.interpolate(output, x.shape[-2:])

        return output
