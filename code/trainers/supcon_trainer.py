# supcon_trainer.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mnist_classifier import MNISTClassifier

class SupConTrainer:
    """
    A minimal trainer for 2-crops SupCon, but
    we handle them as (im1, im2) => each shape [B, C, H, W].
    """
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        device='cuda',
        num_epochs=10,
        patience=3,
        save_path='mnist_supcon.pth',
        resume_path=None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.patience = patience
        self.save_path = save_path

        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        if resume_path is not None and os.path.isfile(resume_path):
            print(f"Resuming training from {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.start_epoch = checkpoint["epoch"] + 1
            self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
            print(f"Resumed at epoch {self.start_epoch}")

    def train(self):
        for epoch in range(1, self.num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate_epoch(epoch)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                torch.save(self.model.state_dict(), self.save_path)
            else:
                self.epochs_no_improve += 1
            
            if self.epochs_no_improve >= self.patience:
                print(f"Early stopping at epoch {epoch}, best val loss: {self.best_val_loss:.4f}")
                break
        print("Training completed.")

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        total = 0

        for (im1, im2), labels in self.train_loader:
            # im1, im2: shape [B, C, H, W]
            # labels: shape [B]
            im1, im2 = im1.to(self.device), im2.to(self.device)
            labels = labels.to(self.device)

            # Combine im1, im2 => shape [2B, C, H, W]
            big_batch = torch.cat([im1, im2], dim=0)
            batch_size = big_batch.size(0)

            # Repeat labels => shape [2B]
            labels_rep = torch.cat([labels, labels], dim=0)

            self.optimizer.zero_grad()
            embeddings = self.model(big_batch)  # shape [2B, latent_dim]
            loss = supcon_loss(embeddings, labels_rep)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * batch_size
            total += batch_size

        avg_loss = running_loss / total
        print(f"Epoch [{epoch}] Train Loss: {avg_loss:.4f}")
        return avg_loss

    def validate_epoch(self, epoch):
        self.model.eval()
        running_loss = 0.0
        total = 0

        with torch.no_grad():
            for (im1, im2), labels in self.val_loader:
                im1, im2 = im1.to(self.device), im2.to(self.device)
                labels = labels.to(self.device)

                big_batch = torch.cat([im1, im2], dim=0)
                batch_size = big_batch.size(0)
                labels_rep = torch.cat([labels, labels], dim=0)

                embeddings = self.model(big_batch)
                val_loss = supcon_loss(embeddings, labels_rep)

                running_loss += val_loss.item() * batch_size
                total += batch_size

        avg_loss = running_loss / total
        print(f"Epoch [{epoch}] Val   Loss: {avg_loss:.4f}")
        return avg_loss

    def classify_evaluation(self, loader, epochs=5):
        """
        For classification eval, we want a single-view loader,
        i.e. shape [B, C, H, W].
        """
        print("Starting linear evaluation...")

        self.model.eval()
        X_list = []
        y_list = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                emb = self.model(images)  # shape [B, latent_dim]
                X_list.append(emb.cpu())
                y_list.append(labels)

        X = torch.cat(X_list, dim=0)
        y = torch.cat(y_list, dim=0)

        # We do a simple linear classifier. We'll define MNISTClassifier or any linear layer
        classifier = MNISTClassifier(latent_dim=X.shape[1], num_classes=10)
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.1)
        criterion = nn.CrossEntropyLoss()

        classifier.train()
        for ep in range(1, epochs+1):
            optimizer.zero_grad()
            logits = classifier(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            print(f"  [LinearEval] epoch {ep}/{epochs}, loss={loss.item():.4f}")

        # measure accuracy
        classifier.eval()
        with torch.no_grad():
            logits = classifier(X)
            preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean().item() * 100
        print(f"Linear Evaluation Accuracy: {acc:.2f}%")
        return acc

##########################
# supcon_loss function
##########################
def supcon_loss(embeddings, labels, temperature=0.07, eps=1e-8):
    """
    Standard supervised contrastive loss: 
      embeddings: shape (2B, d)
      labels: shape (2B,)
    """
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
    loss = -mean_log_prob_pos.mean()
    return loss