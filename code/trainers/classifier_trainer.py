# classifier_trainer.py
import os
import torch
import torch.nn as nn
import torch.optim as optim

class ClassifierTrainer:
    def __init__(self, 
                 encoder, 
                 classifier, 
                 train_loader, 
                 val_loader,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 lr=1e-3, 
                 num_epochs=10,
                 save_path='classifier.pth',
                 resume_path=None):
        """
        Args:
            encoder (nn.Module): Pretrained (frozen) encoder model.
            classifier (nn.Module): Classifier head to train.
            train_loader, val_loader: PyTorch DataLoaders for classification.
            device (str): 'cuda' or 'cpu'.
            lr (float): Learning rate for classifier.
            num_epochs (int): Number of epochs to train.
            save_path (str): Where to save checkpoint each epoch.
            resume_path (str): Optional checkpoint path to resume from.
        """
        self.encoder = encoder.to(device)
        self.classifier = classifier.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.lr = lr
        self.num_epochs = num_epochs
        self.save_path = save_path

        # Freeze the encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.optimizer = optim.Adam(self.classifier.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

        # -- RESUME LOGIC --
        self.start_epoch = 1
        if resume_path is not None and os.path.isfile(resume_path):
            print(f"Resuming training from checkpoint: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)
            self.classifier.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.start_epoch = checkpoint["epoch"] + 1
            print(f"Resumed at epoch {self.start_epoch}")

    def train_epoch(self, epoch):
        self.encoder.eval()
        self.classifier.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            # Forward pass through frozen encoder
            with torch.no_grad():
                latents = self.encoder(images)

            # Forward pass through classifier
            outputs = self.classifier(latents)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        print(f"Epoch [{epoch}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        return train_loss, train_acc

    def validate_epoch(self, epoch):
        self.encoder.eval()
        self.classifier.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                latents = self.encoder(images)
                outputs = self.classifier(latents)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_loss = running_loss / total
        val_acc = correct / total
        print(f"Epoch [{epoch}] - Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f}")
        return val_loss, val_acc

    def train(self):
        """
        Train for self.num_epochs, resuming if start_epoch > 1.
        After each epoch, saves a checkpoint with the current state.
        """
        final_epoch = self.start_epoch + self.num_epochs - 1
        for epoch in range(self.start_epoch, final_epoch + 1):
            self.train_epoch(epoch)
            self.validate_epoch(epoch)

            # -- SAVE CHECKPOINT --
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.classifier.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            }
            torch.save(checkpoint, self.save_path)
            print(f"Checkpoint saved to {self.save_path}")
