# mnist_contrastive.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
sys.path.append('..')
from models.common_autoencoder_blocks import Encoder  
from models.mnist_supconv import MNISTSupCon
from trainers.supcon_trainer import SupConTrainer
from trainers.classifier_trainer import ClassifierTrainer
from models.mnist_classifier import MNISTClassifier


# We want to create two views of the same image
# to train the contrastive loss.
class TwoCropsTransform:
    """
    Apply base_transform twice
    to create two 'views' of the same image.
    """
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        im1 = self.base_transform(x)
        im2 = self.base_transform(x)
        return im1, im2


def run(args):
    base_transform = transforms.Compose([
        transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), shear=5),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2),
        transforms.ToTensor(),
    ])

    train_transform = TwoCropsTransform(base_transform)
    val_base_transform = transforms.Compose([transforms.ToTensor()])
    val_transform = TwoCropsTransform(val_base_transform)

    train_dataset = datasets.MNIST(root=args.data_path, train=True, download=True, transform=train_transform)
    val_dataset   = datasets.MNIST(root=args.data_path, train=False, download=True, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, drop_last=True)

    supcon = MNISTSupCon(
        input_shape=(1,28,28),
        channels=[32,64],
        kernel_sizes=[3,3],
        strides=[2,2],
        paddings=[1,1],
        latent_dim=128,
        hidden_dims=[1024],
        batch_norm_conv=True
    )
    optimizer = torch.optim.Adam(supcon.parameters(), lr=1e-4, weight_decay=1e-4)

    supcon_trainer = SupConTrainer(
        model=supcon,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=args.device,
        num_epochs=20,
        patience=3,
        temperature=0.07
    )
    supcon_trainer.train()
    supcon_trainer.plot_metrics()
    
    # Now we have a trained encoder, we can train a classifier on top of it.
    # We want this classifier to be the same one as in the supervised training.
    # So we need to create a new loader without the TwoCropsTransform that duplicates the images.
    classification_train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor()
    ])
    classification_val_transform = transforms.ToTensor()
    train_dataset_calssification = datasets.MNIST(root=args.data_path, train=True, download=False, transform=classification_train_transform)
    val_dataset_calssification   = datasets.MNIST(root=args.data_path, train=False, download=False, transform=classification_val_transform)
    train_calssification_loader = DataLoader(train_dataset_calssification, batch_size=args.batch_size, shuffle=True)
    val_calssification_loader = DataLoader(val_dataset_calssification, batch_size=args.batch_size, shuffle=False)
    classifier = MNISTClassifier(latent_dim=128, num_classes=10, dropout_fc=0.5, batch_norm_fc=True) 
    classifier_trainer = ClassifierTrainer(
        encoder=supcon,
        classifier=classifier,
        train_loader=train_calssification_loader,
        val_loader=val_calssification_loader,
        device=args.device,
        lr=1e-3,
        weight_decay=1e-4,
        num_epochs=5,
        patience=2,
    )
    classifier_trainer.train()

    print("Contrastive training done.")
    classifier_trainer.plot_metrics()
