# models/mnist_encoder_classifier.py 
# For the supervised classification task (1.2.2)
import torch
import torch.nn as nn
from .mnist_classifier import MNISTClassifier
from .common_autoencoder_blocks import Encoder

class MNISTEncoderClassifier(nn.Module):
    """
    Combines a generic convolutional encoder (from common_autoencoder_blocks)
    with a simple fully-connected classifier head for MNIST classification.
    """
    def __init__(
        self, 
        input_shape=(1,28,28), 
        channels=[32,64], 
        kernel_sizes=[3,3], 
        strides=[2,2], 
        paddings=[1,1], 
        latent_dim=128,
        dropout_conv=0.0,        
        dropout_fc=0.0,
        batch_norm_conv=False,
        batch_norm_fc=False,  
        hidden_dims=None,        
        num_classes=10
    ):
        super(MNISTEncoderClassifier, self).__init__()
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

        self.classifier = MNISTClassifier(latent_dim=latent_dim, num_classes=num_classes, dropout_fc=dropout_fc, batch_norm_fc=batch_norm_fc)

    def forward(self, x):
        z = self.encoder(x)              # encode input -> latent
        logits = self.classifier(z)      # latent -> class scores
        return logits
