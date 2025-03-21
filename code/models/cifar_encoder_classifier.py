import torch.nn as nn
import torch
from .common_autoencoder_blocks import Encoder
from .cifar10_classifier import CIFAR10Classifier

class CIFAR10EncoderClassifier(nn.Module):
    """
    Combines a learnable encoder + classifier in one module,
    for end-to-end training.
    """
    def __init__(self,
                 input_shape=(3, 32, 32),
                 channels=[64, 128, 256],  # maybe deeper for CIFAR10
                 kernel_sizes=[3, 3, 3],
                 strides=[2, 2, 2],
                 paddings=[1, 1, 1],
                 hidden_dims=[1024],
                 dropout_conv=0.0,
                 dropout_fc=0.0,
                 batch_norm_conv=False,
                 batch_norm_fc=False,
                 latent_dim=128,
                 num_classes=10):
        super(CIFAR10EncoderClassifier, self).__init__()
        self.encoder = Encoder(
            input_shape=input_shape,
            channels=channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=paddings,
            latent_dim=latent_dim,
            dropout_conv=dropout_conv,
            batch_norm_conv=batch_norm_conv,
            hidden_dims=hidden_dims
        )
        self.classifier = CIFAR10Classifier(latent_dim=latent_dim, num_classes=num_classes,dropout_fc=dropout_fc, batch_norm_fc=batch_norm_fc)

    def forward(self, x):
        z = self.encoder(x)              # encode input -> latent
        logits = self.classifier(z)      # latent -> class scores
        return logits