# classifier_trainer.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class ClassifierTrainer:
    """
    A trainer for a frozen encoder + trainable classifier scenario.
    Now logs train/val/test accuracy each epoch + final table/plots.
    """
    def __init__(
        self, 
        encoder, 
        classifier, 
        train_loader, 
        val_loader,
        test_loader=None,
        device='cuda',
        lr=1e-3,
        weight_decay=0.0,
        num_epochs=10,
        patience=3,
        save_path=None,
        resume_path=None
    ):
        self.encoder = encoder.to(device)
        self.classifier = classifier.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.patience = patience
        self.save_path = save_path

        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.optimizer = optim.Adam(
            self.classifier.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()

        self.start_epoch = 1
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0

        if resume_path is not None and os.path.isfile(resume_path):
            print(f"Resuming training from checkpoint: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)
            self.classifier.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.start_epoch = checkpoint["epoch"] + 1
            self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
            print(f"Resumed at epoch {self.start_epoch}")

        # We'll store train/val/test accuracy + val_loss
        self.train_acc_history = []
        self.val_acc_history = []
        self.test_acc_history = []
        self.val_loss_history = []

    def train_epoch(self, epoch):
        self.encoder.eval()
        self.classifier.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            with torch.no_grad():
                latents = self.encoder(images)

            self.optimizer.zero_grad()
            outputs = self.classifier(latents)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        avg_loss = running_loss / total
        train_acc = correct / total
        return avg_loss, train_acc

    def evaluate_epoch(self, loader):
        """
        Return (loss, accuracy)
        """
        self.encoder.eval()
        self.classifier.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                latents = self.encoder(images)
                outputs = self.classifier(latents)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

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

            print(f"Epoch [{epoch}] - "
                  f"TrainAcc: {train_acc:.4f}, ValAcc: {val_acc:.4f}", end="")
            if test_acc is not None:
                print(f", TestAcc: {test_acc:.4f}", end="")
            print(f" | ValLoss: {val_loss:.4f}")

            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                if self.save_path:
                    checkpoint = {
                        "epoch": epoch,
                        "model_state_dict": self.classifier.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_val_loss": self.best_val_loss
                    }
                    torch.save(checkpoint, self.save_path)
            else:
                self.epochs_no_improve += 1

            if self.epochs_no_improve >= self.patience:
                print(f"Early stopping triggered at epoch {epoch}")
                print(f"Final val accuracy: {val_acc:.4f}")
                break

        print("Classifier training complete.")

    def plot_metrics(self):
        """
        Plot train/val/test (optional) accuracy for a downstream classifier and validation loss.
        Helps visualize whether the classifier is converging properly or overfitting.
        """
        epochs = range(len(self.train_acc_history))

        # Accuracy plot
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, self.train_acc_history, label='Train Acc', marker='o')
        plt.plot(epochs, self.val_acc_history,   label='Val Acc',   marker='x')
        if self.test_loader is not None and len(self.test_acc_history) == len(epochs):
            plt.plot(epochs, self.test_acc_history, label='Test Acc', marker='s')

        plt.title('Downstream Classifier Accuracy\nAccuracy Trends Over Epochs', fontsize=14)
        plt.suptitle('Shows how accuracy changes for Training, Validation, and Test sets over time.',
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

        plt.title('Classifier Validation Loss Over Epochs', fontsize=14)
        plt.suptitle('Tracks validation loss to detect overfitting or poor convergence.', 
                    fontsize=10, y=0.95)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

        # Print final metrics
        last_idx = len(self.val_loss_history) - 1
        print("=== Final Classifier Metrics ===")
        print(f"Val Loss: {self.val_loss_history[last_idx]:.4f}")
        print(f"Train Acc: {self.train_acc_history[last_idx]*100:.2f}%")
        print(f"Val   Acc: {self.val_acc_history[last_idx]*100:.2f}%")
        if self.test_loader is not None:
            print(f"Test  Acc: {self.test_acc_history[last_idx]*100:.2f}%")
            
