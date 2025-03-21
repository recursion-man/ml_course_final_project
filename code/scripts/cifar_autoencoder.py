import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
sys.path.append('..')
from models.cifar10_autoencoder import CIFAR10Autoencoder
from trainers.autoencoder_trainer import AutoencoderTrainer
from models.cifar10_classifier import CIFAR10Classifier
from trainers.classifier_trainer import ClassifierTrainer

def run(args):
    train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor()
    ])
    val_transform = transforms.ToTensor()
    train_dataset = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=train_transform)
    val_dataset   = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    autoencoder_l1_loss = CIFAR10Autoencoder(latent_dim=128)
    autoencoder_trainer = AutoencoderTrainer(
    model=autoencoder_l1_loss,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=torch.nn.L1Loss(),
    device='cuda',
    lr=5e-4,
    num_epochs=100,
    weight_decay=1e-5,
    )
    autoencoder_trainer.train()
    autoencoder_trainer.plot_metrics()
    classifier2 = CIFAR10Classifier(latent_dim=128, num_classes=10)

    classifier_trainer2 = ClassifierTrainer(
    encoder=autoencoder_l1_loss.encoder,
    classifier=classifier2,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=50,
    patience=5
    )
    classifier_trainer2.train() 
    print("Autoencoder training done.")
    classifier_trainer2.plot_metrics()