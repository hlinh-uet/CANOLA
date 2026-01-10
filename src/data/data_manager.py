"""Data Manager for loading and processing datasets."""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


class DataManager:
    """Manages data loading, processing, and PyTorch tensor preparation."""
    
    def __init__(self, ground_truth_path: str, features_path: str, batch_size=64):
        """
        Initialize DataManager.
        
        Args:
            ground_truth_path: Path to CSV file containing ground truth labels
            features_path: Path to feather file containing embeddings and noisy labels
            batch_size: Batch size for DataLoader
        """
        self.ground_truth_path = ground_truth_path
        self.features_path = features_path
        self.batch_size = batch_size
        
        self.embeddings = None
        self.noisy_labels = None  
        self.true_labels = None   
        self.num_classes = None
        
        # PyTorch tensors
        self.X_tensor = None
        self.y_noisy_tensor = None  # Soft labels (N, C)
        self.y_true_tensor = None

        self._load_and_process_data()
        self._prepare_pytorch_tensors()

    def _load_and_process_data(self):
        """Load and process data from CSV and feather files."""
        print("Starting data loading and processing...")

        # Read ground truth and rename label column
        df_csv = pd.read_csv(self.ground_truth_path)
        df_csv.rename(columns={'label': 'true_label'}, inplace=True)

        # Read features (embeddings & noisy labels)
        df_feather = pd.read_feather(self.features_path)

        # Merge datasets
        df_aligned = pd.concat([df_feather, df_csv[['true_label']]], axis=1)

        self.noisy_labels = df_aligned['label'].values.astype(int)
        self.true_labels = df_aligned['true_label'].values.astype(int)
        
        embedding_df = df_aligned.drop(columns=['label', 'true_label'])
        self.embeddings = embedding_df.values
        
        print(f"✅ Data loaded and processed successfully.")
        print(f"Total samples: {len(self.embeddings)}")
        print(f"Embedding dimension: {self.embeddings.shape[1]}")

    def _prepare_pytorch_tensors(self):
        """Prepare PyTorch tensors from numpy arrays."""
        if self.num_classes is None:
            self.num_classes = len(np.unique(self.true_labels))
            print(f"Number of classes: {self.num_classes}")
        
        self.X_tensor = torch.tensor(self.embeddings, dtype=torch.float32)
        self.y_true_tensor = torch.tensor(self.true_labels, dtype=torch.long)
        
        # Convert hard noisy labels to one-hot (soft labels)
        noisy_labels_one_hot = np.eye(self.num_classes)[self.noisy_labels]
        self.y_noisy_tensor = torch.tensor(noisy_labels_one_hot, dtype=torch.float32)

    def update_noisy_soft_labels(self, new_soft_labels: np.ndarray):
        """
        Update noisy labels with new soft labels for next iteration.
        
        Args:
            new_soft_labels: New soft labels array with shape (N, num_classes)
        """
        print("\n🔄 Updating soft labels for next iteration...")
        if new_soft_labels.shape != (len(self.embeddings), self.num_classes):
            raise ValueError(
                f"Incorrect soft labels shape! "
                f"Expected {(len(self.embeddings), self.num_classes)}, "
                f"got {new_soft_labels.shape}"
            )
        
        # Update tensor with new soft labels
        self.y_noisy_tensor = torch.tensor(new_soft_labels, dtype=torch.float32)
        
        # Update hard labels for evaluation
        self.noisy_labels = np.argmax(new_soft_labels, axis=1)
        print("✅ Soft labels updated successfully!")

    def get_full_dataset(self):
        """Get complete TensorDataset."""
        return TensorDataset(self.X_tensor, self.y_noisy_tensor, self.y_true_tensor)

    def get_full_dataloader(self, shuffle=True):
        """
        Get DataLoader for full dataset.
        
        Args:
            shuffle: Whether to shuffle the data
            
        Returns:
            DataLoader instance
        """
        dataset = self.get_full_dataset()
        return DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=shuffle, 
            drop_last=True
        )

