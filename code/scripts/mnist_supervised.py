# mnist_supervised.py
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
sys.path.append('..')

from models.mnist_encoder_classifier import MNISTEncoderClassifier
from trainers.encoder_classifier_trainer import EncoderClassifierTrainer



def run(args):
    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor()
    ])
    val_transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(root=args.data_path, train=True, download=False, transform=train_transform)
    val_dataset   = datasets.MNIST(root=args.data_path, train=False, download=False, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = MNISTEncoderClassifier(
    input_shape=(1,28,28),
    channels=[32,64],
    kernel_sizes=[3,3],
    strides=[2,2],
    paddings=[1,1],
    latent_dim=128,
    hidden_dims=[1024],
    batch_norm_fc=True,
    batch_norm_conv=True,
    num_classes=10
    )

    trainer = EncoderClassifierTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=args.device,
    lr=1e-3,
    num_epochs=40,
    weight_decay=1e-4,
    early_stopping=True,
    patience=3
    )
    trainer.train()
    print("Supervied training done.")
    trainer.plot_metrics()
    