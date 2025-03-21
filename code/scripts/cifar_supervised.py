import torch
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
sys.path.append('..')

from models.cifar_encoder_classifier import CIFAR10EncoderClassifier
from trainers.encoder_classifier_trainer import EncoderClassifierTrainer



def run(args):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=20),
        transforms.RandomAffine(0, translate=(0.2, 0.2)),
        transforms.ToTensor()
    ])
    val_transform = transforms.ToTensor()

    train_dataset = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=train_transform)
    val_dataset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    model = CIFAR10EncoderClassifier(
    input_shape=(3, 32, 32),
    channels=[64, 128, 256], 
    kernel_sizes=[3, 3, 3],
    strides=[2, 2, 2],
    paddings=[1, 1, 1],
    hidden_dims=[1024],
    dropout_conv=0.0,
    dropout_fc=0.1,
    batch_norm_fc=True,
    batch_norm_conv=True,
    latent_dim=128,
    num_classes=10
    )
    trainer = EncoderClassifierTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device='cuda',
        lr=5e-4,
        num_epochs=60,
        weight_decay=3e-5,
        early_stopping=True,
        patience=5
    )
    trainer.train()
    trainer.plot_metrics()
        