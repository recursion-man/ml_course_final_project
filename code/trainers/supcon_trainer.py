# supcon_trainer.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def supcon_loss(embeddings, labels, temperature=0.07, eps=1e-8):
    embeddings = F.normalize(embeddings, dim=1)
    sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature
    labels = labels.view(-1, 1)
    matches = (labels == labels.T).float()
    diag = torch.eye(matches.shape[0], device=matches.device)
    matches = matches * (1 - diag)

    log_prob = F.log_softmax(sim_matrix, dim=1)
    pos_log_prob = (matches * log_prob).sum(dim=1)
    num_positives = matches.sum(dim=1)
    mean_log_prob_pos = pos_log_prob / (num_positives + eps)
    return -mean_log_prob_pos.mean()

class SupConTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader=None,
        optimizer=None,
        device='cuda',
        num_epochs=10,
        patience=3,
        save_path=None,
        resume_path=None,
        temperature=0.07,
        # NEW: scheduler 
        scheduler_type=None,  # "step", "cosine", "plateau", or None
        step_size=10,
        gamma=0.1,
        T_max=150,
        plateau_factor=0.1,
        plateau_patience=5,
        plateau_threshold=1e-4
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.patience = patience
        self.save_path = save_path
        self.temperature = temperature

        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0

        # histories
        self.train_loss_history = []
        self.val_loss_history   = []
        self.test_loss_history  = []

        # scheduler
        self.scheduler_type = scheduler_type
        self.scheduler = None
        if self.optimizer is not None:
            if self.scheduler_type == "step":
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=step_size, gamma=gamma
                )
            elif self.scheduler_type == "cosine":
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=T_max
                )
            elif self.scheduler_type == "plateau":
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    factor=plateau_factor,
                    patience=plateau_patience,
                    threshold=plateau_threshold
                )

        self.start_epoch = 1
        if resume_path is not None and os.path.isfile(resume_path):
            print(f"Resuming training from {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            if self.optimizer:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.start_epoch = checkpoint["epoch"] + 1
            self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
            print(f"Resumed at epoch {self.start_epoch}")

    def train(self):
        if not self.optimizer:
            raise ValueError("No optimizer provided for SupConTrainer.")

        for epoch in range(self.start_epoch, self.start_epoch + self.num_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss   = self.validate_epoch(epoch)

            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)

            if self.test_loader is not None:
                test_loss = self.evaluate_epoch(self.test_loader)
                self.test_loss_history.append(test_loss)
            else:
                test_loss = None

            # Scheduler step
            if self.scheduler_type == "plateau":
                self.scheduler.step(val_loss)
            elif self.scheduler:
                self.scheduler.step()

            # Early stopping
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

            if self.epochs_no_improve >= self.patience:
                print(f"Early stopping at epoch {epoch}, best val loss: {self.best_val_loss:.4f}")
                break

            print(f"[Epoch {epoch}] TrainLoss={train_loss:.4f}, ValLoss={val_loss:.4f}", end="")
            if test_loss is not None:
                print(f", TestLoss={test_loss:.4f}")
            else:
                print()

        print("SupCon training completed.")

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        total = 0
        for (im1, im2), labels in self.train_loader:
            im1, im2 = im1.to(self.device), im2.to(self.device)
            labels = labels.to(self.device)

            big_batch = torch.cat([im1, im2], dim=0)
            labels_rep = torch.cat([labels, labels], dim=0)

            self.optimizer.zero_grad()
            embeddings = self.model(big_batch)
            loss = supcon_loss(embeddings, labels_rep, self.temperature)
            bs = big_batch.size(0)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * bs
            total += bs

        return running_loss / total

    def validate_epoch(self, epoch):
        return self.evaluate_epoch(self.val_loader)

    def evaluate_epoch(self, loader):
        self.model.eval()
        running_loss = 0.0
        total = 0
        with torch.no_grad():
            for (im1, im2), labels in loader:
                im1, im2 = im1.to(self.device), im2.to(self.device)
                labels = labels.to(self.device)

                big_batch = torch.cat([im1, im2], dim=0)
                labels_rep = torch.cat([labels, labels], dim=0)

                embeddings = self.model(big_batch)
                loss = supcon_loss(embeddings, labels_rep, self.temperature)

                bs = big_batch.size(0)
                running_loss += loss.item() * bs
                total += bs

        return running_loss / total

    def plot_metrics(self):
        epochs = range(len(self.train_loss_history))
        plt.figure(figsize=(8,6))
        plt.plot(epochs, self.train_loss_history, label='Train Loss', marker='o')
        plt.plot(epochs, self.val_loss_history,   label='Val Loss',   marker='x')
        if self.test_loader is not None and len(self.test_loss_history) == len(epochs):
            plt.plot(epochs, self.test_loss_history, label='Test Loss', marker='s')

        plt.title('SupCon Training Curve', fontsize=14)
        plt.suptitle('Visualizing SupCon training/val/test loss over epochs.',
                     fontsize=10, y=0.95)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Contrastive Loss', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

        # final
        print("=== Final SupCon Metrics ===")
        last_idx = len(self.train_loss_history) - 1
        print(f"Train Loss: {self.train_loss_history[last_idx]:.4f}")
        print(f"Val   Loss: {self.val_loss_history[last_idx]:.4f}")
        if self.test_loader is not None:
            print(f"Test  Loss: {self.test_loss_history[last_idx]:.4f}")
