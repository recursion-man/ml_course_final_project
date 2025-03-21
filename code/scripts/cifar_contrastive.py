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
from models.cifar10_supconv import CIFAR10SupCon
from trainers.supcon_trainer import SupConTrainer
from trainers.classifier_trainer import ClassifierTrainer
from models.cifar10_classifier import CIFAR10Classifier
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class TwoCropsTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        im1 = self.base_transform(x)
        im2 = self.base_transform(x)
        return im1, im2


def run(args):
    base_transform = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), shear=5),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2),
    transforms.ToTensor(),
    ])

    train_transform = TwoCropsTransform(base_transform)
    val_base_transform = transforms.Compose([transforms.ToTensor()])
    val_transform = TwoCropsTransform(val_base_transform)

    train_dataset = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=train_transform)
    val_dataset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, drop_last=True)

    supcon = CIFAR10SupCon(
        input_shape=(3, 32, 32),
        channels=[64, 128,256,512],  # maybe deeper for CIFAR10
        kernel_sizes=[3, 3, 3,3],
        strides=[2, 2, 2,2],
        paddings=[1, 1, 1,1],
        latent_dim=128,
        dropout_rate=0.1,
        hidden_dims=[1024],
        batch_norm_conv=True
    )
    optimizer = torch.optim.Adam(supcon.parameters(), lr=1e-4, weight_decay=1e-4)

    supcon_trainer = SupConTrainer(
        model=supcon,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=60,
        patience=5,
        temperature=0.2
    )
    supcon_trainer.train()
    supcon_trainer.plot_metrics()
    
    classifier_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), shear=5),
    transforms.ToTensor(),
])

    classifier_val_transform = val_base_transform  


    train_dataset_calssification = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=classifier_train_transform)
    val_dataset_calssification   = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=classifier_val_transform)
    train_calssification_loader = DataLoader(train_dataset_calssification, batch_size=128, shuffle=True)
    val_calssification_loader = DataLoader(val_dataset_calssification, batch_size=128, shuffle=False)
    classifier = CIFAR10Classifier(latent_dim=128, num_classes=10, dropout_fc = 0.2, batch_norm_fc=True) 
    classifier_trainer = ClassifierTrainer(
        encoder=supcon,
        classifier=classifier,
        train_loader=train_calssification_loader,
        val_loader=val_calssification_loader,
        device=device,
        lr=5e-5,
        weight_decay=3e-4,
        num_epochs=60,
        patience=10,
    )
    classifier_trainer.train()
    classifier_trainer.plot_metrics()