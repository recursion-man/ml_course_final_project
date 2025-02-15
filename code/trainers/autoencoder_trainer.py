# trainers/autoencoder_trainer.py
import torch
import torch.nn as nn
import torch.optim as optim

class AutoencoderTrainer:
    def __init__(self, 
                 model, 
                 train_loader, 
                 val_loader,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 lr=1e-3, 
                 num_epochs=10,
                 save_path='autoencoder.pth',
                 resume_path=None,
                 weight_decay=0.0, 
                 early_stopping=False,
                 patience=3):
        """
        Simple trainer for self-supervised autoencoder training.
        No learning rate scheduling.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.lr = lr
        self.num_epochs = num_epochs
        self.save_path = save_path
        
        # We can use weight decay if we want
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()  # reconstruction loss

        self.early_stopping = early_stopping
        self.patience = patience
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0

        self.start_epoch = 1
        if resume_path is not None:
            checkpoint = torch.load(resume_path, map_location=device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.start_epoch = checkpoint["epoch"] + 1
            self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
            print(f"Resumed training from {resume_path}, starting epoch {self.start_epoch}")

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        for data, _ in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            reconstruction, _ = self.model(data)
            loss = self.criterion(reconstruction, data)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * data.size(0)
        avg_loss = total_loss / len(self.train_loader.dataset)
        print(f"Epoch [{epoch}] - Train Loss: {avg_loss:.4f}")
        return avg_loss

    def validate_epoch(self, epoch):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for data, _ in self.val_loader:
                data = data.to(self.device)
                reconstruction, _ = self.model(data)
                loss = self.criterion(reconstruction, data)
                total_loss += loss.item() * data.size(0)
        avg_loss = total_loss / len(self.val_loader.dataset)
        print(f"Epoch [{epoch}] - Val   Loss: {avg_loss:.4f}")
        return avg_loss

    def train(self):
        final_epoch = self.start_epoch + self.num_epochs - 1
        for epoch in range(self.start_epoch, final_epoch + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate_epoch(epoch)

            # checkpoint
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_loss": self.best_val_loss
            }

            # check improvement for early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                torch.save(checkpoint, self.save_path)
            else:
                self.epochs_no_improve += 1

            if self.early_stopping and self.epochs_no_improve >= self.patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        print("Autoencoder training complete.")
