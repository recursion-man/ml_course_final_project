# models/mnist_classifier.py
# A classifier for 1.2.1 (autoencoder) and 1.2.3 (contrastive)
import torch.nn as nn

class MNISTClassifier(nn.Module):
    """
    Classifier for 128-dim MNIST latent vector -> 10 classes
    """
    def __init__(self, latent_dim=128, num_classes=10, dropout_fc=0.0, batch_norm_fc=False):
        super(MNISTClassifier, self).__init__()
        layers = []
        layers.append(nn.Linear(latent_dim, 64))
        if batch_norm_fc:
            layers.append(nn.BatchNorm1d(64))
        layers.append(nn.ReLU(True))
        layers.append(nn.Dropout(p=dropout_fc))
        layers.append(nn.Linear(64, num_classes))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)
