# autoencoder_trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class AutoencoderTrainer:
    def __init__(
        self, 
        model, 
        train_loader, 
        val_loader,
        test_loader=None,   # optional test
        criterion=None,     # MSELoss or L1Loss
        device='cuda',
        lr=1e-3, 
        num_epochs=10,
        save_path=None,
        resume_path=None,
        weight_decay=0.0, 
        early_stopping=True,
        patience=3,
        scheduler_type=None,  # "step" or "cosine" or None
        step_size=30,         # for StepLR
        gamma=0.1,            # for StepLR
        T_max=150             # for CosineAnnealingLR
    ):
        """
        Autoencoder trainer with optional LR scheduler support.
        If scheduler_type=None, no LR schedule is used (same as old code).
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.lr = lr
        self.num_epochs = num_epochs
        self.save_path = save_path
        
        # 1) Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=weight_decay
        )

        self.criterion = criterion if criterion else nn.L1Loss()
        self.l1_criterion = nn.L1Loss()  
        self.early_stopping = early_stopping
        self.patience = patience
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0


        self.scheduler_type = scheduler_type
        self.scheduler = None
        if self.scheduler_type == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma
            )
        elif self.scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=T_max
            )
        elif self.scheduler_type == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.1, patience=5
            )    

        self.start_epoch = 1
        if resume_path is not None:
            checkpoint = torch.load(resume_path, map_location=device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.start_epoch = checkpoint["epoch"] + 1
            self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
            print(f"Resumed training from {resume_path}, starting epoch {self.start_epoch}")

        # 5) Histories
        self.train_loss_history = []
        self.val_loss_history   = []
        self.test_loss_history  = []

        self.train_l1_history  = []
        self.val_l1_history    = []
        self.test_l1_history   = []

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_l1   = 0.0
        count      = 0

        for data, _ in self.train_loader:
            data = data.to(self.device)

            self.optimizer.zero_grad()
            reconstruction, _ = self.model(data)
            loss = self.criterion(reconstruction, data)
            loss.backward()
            self.optimizer.step()

            bs = data.size(0)
            total_loss += loss.item() * bs

            l1_loss   = self.l1_criterion(reconstruction, data)
            total_l1 += l1_loss.item() * bs
            count    += bs

        avg_loss = total_loss / count
        avg_l1   = total_l1 / count
        return avg_loss, avg_l1

    def evaluate_epoch(self, loader):
        self.model.eval()
        total_loss = 0.0
        total_l1   = 0.0
        count      = 0

        with torch.no_grad():
            for data, _ in loader:
                data = data.to(self.device)
                reconstruction, _ = self.model(data)

                loss   = self.criterion(reconstruction, data)
                l1loss = self.l1_criterion(reconstruction, data)

                bs          = data.size(0)
                total_loss += loss.item() * bs
                total_l1   += l1loss.item() * bs
                count      += bs

        avg_loss = total_loss / count
        avg_l1   = total_l1 / count
        return avg_loss, avg_l1

    def train(self):
        final_epoch = self.start_epoch + self.num_epochs - 1

        for epoch in range(self.start_epoch, final_epoch + 1):

            train_loss, train_l1 = self.train_epoch(epoch)
            val_loss,   val_l1   = self.evaluate_epoch(self.val_loader)

            if self.test_loader is not None:
                test_loss, test_l1 = self.evaluate_epoch(self.test_loader)
            else:
                test_loss, test_l1 = (None, None)

            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)
            if test_loss is not None:
                self.test_loss_history.append(test_loss)

            self.train_l1_history.append(train_l1)
            self.val_l1_history.append(val_l1)
            if test_l1 is not None:
                self.test_l1_history.append(test_l1)


            print(f"Epoch [{epoch}] - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}", end="")
            if test_loss is not None:
                print(f", Test Loss: {test_loss:.4f}", end="")
            print()
            print(f"              Train L1: {train_l1:.4f},  Val L1: {val_l1:.4f}", end="")
            if test_l1 is not None:
                print(f", Test L1: {test_l1:.4f}", end="")
            print()

            if self.scheduler_type == "plateau":
                self.scheduler.step(val_loss)
            elif self.scheduler is not None:
                self.scheduler.step()

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                if self.save_path is not None:
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
                break

        print("Autoencoder training complete.")

    def plot_metrics(self):
        """
        Plot reconstruction losses (train/val/test if available) and L1 errors (train/val/test).
        If test_loader was used, it will also be plotted.
        """
        epochs = range(len(self.train_loss_history))

        # Plot reconstruction loss
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, self.train_loss_history, label='Train Loss', marker='o')
        plt.plot(epochs, self.val_loss_history,   label='Val Loss',   marker='x')
        if self.test_loader is not None and len(self.test_loss_history) == len(epochs):
            plt.plot(epochs, self.test_loss_history, label='Test Loss', marker='s')

        plt.title('Autoencoder Reconstruction Loss', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

        # Plot L1 error
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, self.train_l1_history, label='Train L1', marker='o')
        plt.plot(epochs, self.val_l1_history,   label='Val L1',   marker='x')
        if self.test_loader is not None and len(self.test_l1_history) == len(epochs):
            plt.plot(epochs, self.test_l1_history, label='Test L1', marker='s')

        plt.title('Autoencoder Mean Absolute Error', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('L1 Error', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

        # Print final
        last_idx = len(self.train_loss_history) - 1
        print("=== Final Autoencoder Metrics ===")
        print(f"Train ReconLoss: {self.train_loss_history[last_idx]:.4f}")
        print(f"Val   ReconLoss: {self.val_loss_history[last_idx]:.4f}")
        if self.test_loader is not None:
            print(f"Test  ReconLoss: {self.test_loss_history[last_idx]:.4f}")
        print(f"Train L1: {self.train_l1_history[last_idx]:.4f}")
        print(f"Val   L1: {self.val_l1_history[last_idx]:.4f}")
        if self.test_loader is not None:
            print(f"Test  L1: {self.test_l1_history[last_idx]:.4f}")
