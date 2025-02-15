import torch.nn as nn
import torch
from .common_autoencoder_blocks import Encoder
from .cifar10_classifier import CIFAR10Classifier

class CIFAR10EncoderClassifier(nn.Module):
    """
    Combines a learnable encoder + classifier in one module,
    for end-to-end training.
    """
    def __init__(self, latent_dim=128, num_classes=10):
        super(CIFAR10EncoderClassifier, self).__init__()
        self.encoder = Encoder(
            input_shape=(3,32,32),
            channels=[32,64,128],  # maybe deeper for CIFAR10
            kernel_sizes=[3,3,3],
            strides=[2,2,2],
            paddings=[1,1,1],
            latent_dim=latent_dim
        )
        self.classifier = CIFAR10Classifier(latent_dim=latent_dim, num_classes=num_classes)

    def forward(self, x):
        z = self.encoder(x)              # encode input -> latent
        logits = self.classifier(z)      # latent -> class scores
        return logits
