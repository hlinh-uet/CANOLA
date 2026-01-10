"""Asymmetric Co-Training (ACT) Trainer."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from ..models import MLP


class ACTTrainer:
    """
    Asymmetric Co-Training Trainer for robust model learning.
    
    Uses two models:
    - RTM (Robust Training Model): Trained on clean samples
    - NTM (Noisy Training Model): Trained on all samples
    """
    
    def __init__(self, input_dim, num_classes, hidden_dims=[512, 256], 
                 lr_rtm=1e-4, lr_ntm=1e-3):
        """
        Initialize ACT Trainer.
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of classes
            hidden_dims: Hidden layer dimensions
            lr_rtm: Learning rate for RTM
            lr_ntm: Learning rate for NTM
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        
        self.rtm = MLP(input_dim, num_classes, hidden_dims).to(self.device)
        self.ntm = MLP(input_dim, num_classes, hidden_dims).to(self.device)
        
        self.optimizer_rtm = optim.Adam(self.rtm.parameters(), lr=lr_rtm)
        self.optimizer_ntm = optim.Adam(self.ntm.parameters(), lr=lr_ntm)
        
        self.criterion = nn.CrossEntropyLoss()

    def train(self, dataloader, epochs=100, warmup_epochs=20, early_stopper=None):
        """
        Train models using ACT strategy.
        
        Args:
            dataloader: DataLoader for training
            epochs: Total number of epochs
            warmup_epochs: Number of warmup epochs (both models trained on all data)
            early_stopper: Early stopping callback
            
        Returns:
            Trained RTM (robust model)
        """
        full_dataset = dataloader.dataset
        X_full = full_dataset.tensors[0].to(self.device)
        y_noisy_full_soft = full_dataset.tensors[1].to(self.device)  # Soft labels (N, C)

        for epoch in range(epochs):
            self.rtm.train()
            self.ntm.train()
            
            # Warmup phase: train both models on all data
            if epoch < warmup_epochs:
                for features, noisy_labels_soft, _ in dataloader:
                    features = features.to(self.device)
                    noisy_labels_soft = noisy_labels_soft.to(self.device)
                    
                    # Update RTM
                    self.optimizer_rtm.zero_grad()
                    loss_rtm = self.criterion(self.rtm(features), noisy_labels_soft)
                    loss_rtm.backward()
                    self.optimizer_rtm.step()
                    
                    # Update NTM
                    self.optimizer_ntm.zero_grad()
                    loss_ntm = self.criterion(self.ntm(features), noisy_labels_soft)
                    loss_ntm.backward()
                    self.optimizer_ntm.step()
                
                if (epoch + 1) % 5 == 0:
                    print(f"Warmup Epoch [{epoch+1}/{warmup_epochs}]")
                    
            # ACT phase: RTM on clean samples, NTM on all samples
            else:
                self.rtm.eval()
                self.ntm.eval()
                
                with torch.no_grad():
                    # Get hard predictions for comparison
                    preds_rtm_hard = torch.argmax(self.rtm(X_full), dim=1)
                    preds_ntm_hard = torch.argmax(self.ntm(X_full), dim=1)
                
                # Convert soft labels to hard for consensus checking
                y_noisy_hard = torch.argmax(y_noisy_full_soft, dim=1)

                # Find samples where both models agree with noisy label
                agree_mask = (preds_rtm_hard == y_noisy_hard) & (preds_ntm_hard == y_noisy_hard)
                
                # Mining phase: include samples where NTM agrees but RTM doesn't
                mine_mask = torch.zeros_like(agree_mask)
                if epoch < warmup_epochs + (epochs - warmup_epochs) / 2:
                    mine_mask = (preds_rtm_hard != y_noisy_hard) & (preds_ntm_hard == y_noisy_hard)
                
                clean_indices_mask = agree_mask | mine_mask
                num_clean = clean_indices_mask.sum().item()

                if num_clean == 0:
                    print(f"Epoch [{epoch+1}/{epochs}]: No clean samples found, skipping RTM update.")
                    # Only train NTM on all samples
                    self.ntm.train()
                    for features, noisy_labels_soft, _ in dataloader:
                        features = features.to(self.device)
                        noisy_labels_soft = noisy_labels_soft.to(self.device)
                        self.optimizer_ntm.zero_grad()
                        loss_ntm = self.criterion(self.ntm(features), noisy_labels_soft)
                        loss_ntm.backward()
                        self.optimizer_ntm.step()
                    continue
                
                # Create clean dataset
                clean_dataset = TensorDataset(
                    X_full[clean_indices_mask], 
                    y_noisy_full_soft[clean_indices_mask]
                )
                clean_dataloader = DataLoader(
                    clean_dataset, 
                    batch_size=dataloader.batch_size, 
                    shuffle=True, 
                    drop_last=True
                )
                
                self.rtm.train()
                self.ntm.train()
                
                clean_iter = iter(clean_dataloader)
                
                for features, noisy_labels_soft, _ in dataloader:
                    features = features.to(self.device)
                    noisy_labels_soft = noisy_labels_soft.to(self.device)
                    
                    # Update RTM on clean samples
                    try:
                        clean_features, clean_labels_soft = next(clean_iter)
                        self.optimizer_rtm.zero_grad()
                        loss_rtm = self.criterion(self.rtm(clean_features), clean_labels_soft)
                        loss_rtm.backward()
                        self.optimizer_rtm.step()
                    except StopIteration:
                        pass

                    # Update NTM on all samples
                    self.optimizer_ntm.zero_grad()
                    loss_ntm = self.criterion(self.ntm(features), noisy_labels_soft)
                    loss_ntm.backward()
                    self.optimizer_ntm.step()

                if (epoch + 1) % 10 == 0:
                    print(f"ACT Epoch [{epoch+1}/{epochs}], Clean samples selected: {num_clean}/{len(X_full)}")

            # Early stopping check
            if early_stopper:
                self.ntm.eval()
                total_ntm_loss = 0
                with torch.no_grad():
                    eval_loader = DataLoader(
                        full_dataset, 
                        batch_size=dataloader.batch_size, 
                        shuffle=False
                    )
                    for features, noisy_labels_soft, _ in eval_loader:
                        features = features.to(self.device)
                        noisy_labels_soft = noisy_labels_soft.to(self.device)
                        loss_ntm = self.criterion(self.ntm(features), noisy_labels_soft)
                        total_ntm_loss += loss_ntm.item()
                avg_ntm_loss = total_ntm_loss / len(eval_loader)
                
                print(f"Epoch [{epoch+1}/{epochs}], NTM Loss (for early stopping): {avg_ntm_loss:.4f}")
                if early_stopper(avg_ntm_loss):
                    break
        
        print("Co-training completed")
        return self.rtm

