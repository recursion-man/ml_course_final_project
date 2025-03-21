import torch
import torch.nn as nn
from .common_autoencoder_blocks import Encoder

class CIFAR10SupCon(nn.Module):
    """
    A wrapper that uses the Encoder for MNIST,
    returning embeddings suitable for SupCon training.
    """
    def __init__(
        self,
        input_shape=(3, 32, 32),
        channels=[64, 128, 256],  # maybe deeper for CIFAR10
        kernel_sizes=[3, 3, 3],
        strides=[2, 2, 2],
        paddings=[1, 1, 1],
        latent_dim=128,
        dropout_conv=0.0,
        dropout_fc=0.0,
        dropout_rate=0.0,
        batch_norm_conv=False,
        hidden_dims=None
    ):
        super(CIFAR10SupCon, self).__init__()
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
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None
    def forward(self, x):
        z = self.encoder(x)
        if self.dropout is not None:
            z = self.dropout(z)
        return z