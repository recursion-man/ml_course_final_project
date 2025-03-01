# encoder_classifier_trainer.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class EncoderClassifierTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader=None,
        device='cuda',
        lr=1e-3,
        num_epochs=10,
        save_path=None,
        resume_path=None,
        weight_decay=0.0,
        early_stopping=True,
        patience=3
    ):
        """
        End-to-end classification (encoder+classifier).
        We'll track train/val/test accuracy, plus val loss for early stopping.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.lr = lr
        self.num_epochs = num_epochs
        self.save_path = save_path

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
        # For storing metrics
        self.train_acc_history = []
        self.val_acc_history = []
        self.test_acc_history = []
        self.val_loss_history = []

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

            bs = images.size(0)
            running_loss += loss.item() * bs
            _, predicted = torch.max(logits, dim=1)
            correct += (predicted == labels).sum().item()
            total += bs

        avg_loss = running_loss / total
        train_acc = correct / total
        print(f"Epoch [{epoch}] - Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}")
        return avg_loss, train_acc

    def evaluate_epoch(self, loader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                logits = self.model(images)
                loss = self.criterion(logits, labels)

                bs = images.size(0)
                running_loss += loss.item() * bs
                _, predicted = torch.max(logits, dim=1)
                correct += (predicted == labels).sum().item()
                total += bs

        avg_loss = running_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    def train(self):
        final_epoch = self.start_epoch + self.num_epochs - 1
        for epoch in range(self.start_epoch, final_epoch + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.evaluate_epoch(self.val_loader)
            if self.test_loader is not None:
                test_loss, test_acc = self.evaluate_epoch(self.test_loader)
            else:
                test_loss, test_acc = (None, None)

            self.train_acc_history.append(train_acc)
            self.val_acc_history.append(val_acc)
            if test_acc is not None:
                self.test_acc_history.append(test_acc)
            self.val_loss_history.append(val_loss)

            print(f"Epoch [{epoch}] - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}", end="")
            if test_acc is not None:
                print(f", Test Acc: {test_acc:.4f}", end="")
            print()

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                if self.save_path:
                    checkpoint = {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_val_loss": self.best_val_loss
                    }
                    torch.save(checkpoint, self.save_path)
            else:
                self.epochs_no_improve += 1

            if self.early_stopping and self.epochs_no_improve >= self.patience:
                print(f"Early stopping triggered at epoch {epoch}")
                print(f"Final val accuracy: {val_acc:.4f}")
                break

        print("Classification training complete.")

    def plot_metrics(self):
        """
        Plots the accuracy for train/val/test (if available) for an encoder+classifier setup,
        as well as the validation loss over epochs.
        """
        epochs = range(len(self.train_acc_history))

        # Accuracy plot
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, self.train_acc_history, label='Train Acc', marker='o')
        plt.plot(epochs, self.val_acc_history,   label='Val Acc',   marker='x')
        if self.test_loader is not None and len(self.test_acc_history) == len(epochs):
            plt.plot(epochs, self.test_acc_history, label='Test Acc', marker='s')

        plt.title('Encoder+Classifier Accuracy\nAccuracy Trends Over Epochs', fontsize=14)
        plt.suptitle('Shows how the accuracy changes for Training and Validation.',
                    fontsize=10, y=0.95)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

        # Validation loss plot
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, self.val_loss_history, label='Val Loss', marker='o', color='red')

        plt.title('Encoder+Classifier Validation Loss Over Epochs', fontsize=14)
        plt.suptitle('Tracks the validation loss to inspect overfitting or underfitting trends.', 
                    fontsize=10, y=0.95)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

        # Print final metrics
        last_idx = len(self.val_loss_history) - 1
        print("=== Final Encoder+Classifier Metrics ===")
        print(f"Val Loss: {self.val_loss_history[last_idx]:.4f}")
        print(f"Train Acc: {self.train_acc_history[last_idx]*100:.2f}%")
        print(f"Val   Acc: {self.val_acc_history[last_idx]*100:.2f}%")
        if self.test_loader is not None:
            print(f"Test  Acc: {self.test_acc_history[last_idx]*100:.2f}%")
