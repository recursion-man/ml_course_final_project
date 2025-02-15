# models/cifar10_classifier.py
import torch.nn as nn

class CIFAR10Classifier(nn.Module):
    """
    Classifier for 128-dim CIFAR10 latent vector -> 10 classes
    """
    def __init__(self, latent_dim=128, num_classes=10):
        super(CIFAR10Classifier, self).__init__()
        # Possibly bigger net than MNIST
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)
