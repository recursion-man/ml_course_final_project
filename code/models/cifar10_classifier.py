# models/cifar10_classifier.py
import torch.nn as nn

class CIFAR10Classifier(nn.Module):
    """
    Classifier for 128-dim CIFAR10 latent vector -> 10 classes
    """

    def __init__(self, latent_dim=128, num_classes=10, dropout_fc = 0.2, batch_norm_fc=True):
        super(CIFAR10Classifier, self).__init__()
        layers = []
        layers.append(nn.Linear(latent_dim, 256))
        if batch_norm_fc:
            layers.append(nn.BatchNorm1d(256))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(p=dropout_fc))

        layers.append(nn.Linear(256, 128))
        if batch_norm_fc:
            layers.append(nn.BatchNorm1d(128))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(p=dropout_fc))

        layers.append(nn.Linear(128, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)