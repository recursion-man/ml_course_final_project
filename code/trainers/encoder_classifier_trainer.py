# trainers/encoder_classifier_trainer.py
import os
import torch
import torch.nn as nn
import torch.optim as optim

class EncoderClassifierTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        lr=1e-3,
        num_epochs=10,
        save_path='encoder_classifier.pth',
        resume_path=None,
        weight_decay=0.0,
        early_stopping=True,
        patience=3
    ):
        """
        A trainer for end-to-end classification (encoder+classifier),
        with no LR scheduler.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.lr = lr
        self.num_epochs = num_epochs
        self.save_path = save_path

        # Use weight decay for regularization if needed
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()

        self.early_stopping = early_stopping
        self.patience = patience
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0

        self.start_epoch = 1
        # Resume
        if resume_path is not None and os.path.isfile(resume_path):
            print(f"Resuming training from {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.start_epoch = checkpoint["epoch"] + 1
            self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
            print(f"Resumed at epoch {self.start_epoch}")

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(logits, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        avg_loss = running_loss / total
        accuracy = correct / total
        print(f"Epoch [{epoch}] - Train Loss: {avg_loss:.4f}, Train Acc: {accuracy:.4f}")
        return avg_loss, accuracy

    def validate_epoch(self, epoch):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                logits = self.model(images)
                loss = self.criterion(logits, labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(logits, dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        avg_loss = running_loss / total
        accuracy = correct / total
        print(f"Epoch [{epoch}] - Val   Loss: {avg_loss:.4f}, Val   Acc: {accuracy:.4f}")
        return avg_loss, accuracy

    def train(self):
        final_epoch = self.start_epoch + self.num_epochs - 1
        for epoch in range(self.start_epoch, final_epoch + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate_epoch(epoch)

            # Checkpoint
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_loss": self.best_val_loss
            }

            # Check if improved
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                torch.save(checkpoint, self.save_path)
            else:
                self.epochs_no_improve += 1

            # Early stopping
            if self.early_stopping and self.epochs_no_improve >= self.patience:
                print(f"Early stopping triggered at epoch {epoch} \n End val loss : {self.best_val_loss:.4f}")
                break

        print("Classification training complete.")
