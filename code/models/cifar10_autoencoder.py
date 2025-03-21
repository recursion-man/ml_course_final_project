# models/cifar10_autoencoder.py
import torch.nn as nn
import torch
from .common_autoencoder_blocks import Encoder, Decoder

class CIFAR10Autoencoder(nn.Module):
    """
    Example autoencoder specifically for CIFAR10 (3 x 32 x 32).
    Potentially deeper or wider than MNIST version.
    """
    def __init__(self, latent_dim=128):
        super(CIFAR10Autoencoder, self).__init__()
        # This might differ from the MNIST version
        self.encoder = Encoder(
            input_shape=(3,32,32),
            channels=[64,128,256],  # maybe deeper for CIFAR10
            kernel_sizes=[3,3,3],
            strides=[2,2,2],
            paddings=[1,1,1],
            latent_dim=latent_dim,
            batch_norm_conv = True
        )
        self.decoder = Decoder(
            conv_output_shape=self.encoder.conv_output_shape,
            channels=[256,128,64],
            kernel_sizes=[3,3,3],
            strides=[2,2,2],
            paddings=[1,1,1],
            latent_dim=latent_dim,
            output_channels=3,
            batch_norm_conv = True,

        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent