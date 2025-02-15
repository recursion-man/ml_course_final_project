# models/mnist_autoencoder.py
import torch.nn as nn
import torch
from .common_autoencoder_blocks import Encoder, Decoder  # Suppose you keep shared blocks in a "common_autoencoder_blocks.py"

class MNISTAutoencoder(nn.Module):
    """
    Example autoencoder specifically for MNIST (1 x 28 x 28).
    """
    def __init__(self, latent_dim=128):
        super(MNISTAutoencoder, self).__init__()
        # For instance, we can pass param lists specifically suited for MNIST
        self.encoder = Encoder(
            input_shape=(1,28,28),
            channels=[32,64],
            kernel_sizes=[3,3],
            strides=[2,2],
            paddings=[1,1],
            latent_dim=latent_dim
        )
        self.decoder = Decoder(
            conv_output_shape=self.encoder.conv_output_shape,
            channels=[32,64],
            kernel_sizes=[3,3],
            strides=[2,2],
            paddings=[1,1],
            latent_dim=latent_dim,
            output_channels=1
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent