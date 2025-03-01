# models/mnist_supconv.py 
# For the unsupervised contrastive learning task (1.2.3)
import torch
import torch.nn as nn
from .common_autoencoder_blocks import Encoder

class MNISTSupCon(nn.Module):
    """
    A wrapper that uses the Encoder for MNIST,
    returning embeddings suitable for SupCon training.
    """
    def __init__(
        self,
        input_shape=(1,28,28),
        channels=[32, 64],
        kernel_sizes=[3, 3],
        strides=[2, 2],
        paddings=[1, 1],
        latent_dim=128,
        dropout_conv=0.0,
        batch_norm_conv=False,
        hidden_dims=None
    ):
        super(MNISTSupCon, self).__init__()
        self.encoder = Encoder(
            input_shape=input_shape,
            channels=channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=paddings,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,  # in the article they used 2048
            dropout_conv=dropout_conv,
            batch_norm_conv=batch_norm_conv
        )

    def forward(self, x):
        z = self.encoder(x)
        return z