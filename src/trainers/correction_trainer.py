"""Correction Trainer for fine-tuning with forward correction loss."""

import torch
from torch.utils.data import DataLoader


class CorrectionTrainer:
    """
    Trainer for fine-tuning model with forward correction loss.
    """
    
    def __init__(self, model, optimizer, loss_fn, device):
        """
        Initialize Correction Trainer.
        
        Args:
            model: PyTorch model to train
            optimizer: Optimizer for training
            loss_fn: Loss function (typically ForwardCorrectionLoss)
            device: Device to train on (cpu/cuda)
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def train(self, dataloader, epochs=80, early_stopper=None):
        """
        Train model with correction loss.
        
        Args:
            dataloader: DataLoader for training
            epochs: Number of training epochs
            early_stopper: Early stopping callback
            
        Returns:
            Tuple of (trained_model, final_average_loss)
        """
        print(f"Starting fine-tuning (max {epochs} epochs)...")
        self.model.train()
        final_avg_loss = 0
        
        for epoch in range(epochs):
            total_loss = 0
            for features, noisy_labels, _ in dataloader:
                features = features.to(self.device)
                noisy_labels = noisy_labels.to(self.device)
                
                self.optimizer.zero_grad()
                clean_logits = self.model(features)
                loss = self.loss_fn(clean_logits, noisy_labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            final_avg_loss = avg_loss
            print(f"Epoch [{epoch+1}/{epochs}], Correction Loss: {avg_loss:.4f}")
            
            if early_stopper and early_stopper(avg_loss):
                break
        
        print("Fine-tuning completed.")
        return self.model, final_avg_loss

