# mnist_autoencoder.py
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
sys.path.append('..')
from models.mnist_autoencoder import MNISTAutoencoder
from trainers.autoencoder_trainer import AutoencoderTrainer
from models.mnist_classifier import MNISTClassifier
from trainers.classifier_trainer import ClassifierTrainer

def run(args):
    train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=5),
    transforms.RandomAffine(0, translate=(0.05, 0.05)),
    transforms.ToTensor()
    ])
    val_transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root=args.data_path, train=True, download=True, transform=train_transform)
    val_dataset   = datasets.MNIST(root=args.data_path, train=False, download=True, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    autoencoder = MNISTAutoencoder(latent_dim=128)
    autoencoder_trainer = AutoencoderTrainer(
    model=autoencoder,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=torch.nn.L1Loss(),
    device='cuda',
    lr=1e-4,
    num_epochs=60,
    weight_decay=1e-4,
    )
    autoencoder_trainer.train()

    autoencoder_trainer = AutoencoderTrainer(
    model=autoencoder,
    train_loader=train_loader,
    val_loader=val_loader,
    device=args.device,
    lr=1e-3,
    num_epochs=40,
    patience=3
    )
    print("Training autoencoder...")
    autoencoder_trainer.train()
    print("Finished training autoencoder...")
    autoencoder_trainer.plot_metrics()

    classifier = MNISTClassifier(latent_dim=128, num_classes=10)
    classifier_trainer = ClassifierTrainer(
    encoder=autoencoder.encoder,
    classifier=classifier,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=40,
    patience=3
    )
    classifier_trainer.train() 

    print("Autoencoder training done.")

    classifier_trainer.plot_metrics()

