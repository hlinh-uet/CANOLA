"""Training utilities."""

import random
import numpy as np
import torch


def set_seed(seed):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopper:
    """Early stopping callback to stop training when loss doesn't improve."""
    
    def __init__(self, patience=10, min_delta=0.0001):
        """
        Initialize Early Stopper.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change in loss to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, loss):
        """
        Check if training should stop.
        
        Args:
            loss: Current loss value
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_loss - loss > self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"🛑 Early stopping! Loss has not improved for {self.patience} epochs.")
                return True
        return False

