"""Evaluation utilities for noise correction."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch


def calculate_ground_truth_T(true_labels, noisy_labels, num_classes):
    """
    Calculate ground truth noise transition matrix from true and noisy labels.
    
    T[i,j] = P(noisy=j | clean=i)
    
    Args:
        true_labels: Ground truth labels (N,)
        noisy_labels: Noisy labels (N,)
        num_classes: Number of classes
        
    Returns:
        Ground truth transition matrix (num_classes, num_classes)
    """
    T_true = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        # Find samples with true label i
        indices = np.where(true_labels == i)[0]
        if len(indices) == 0:
            continue
        
        # Get corresponding noisy labels
        noisy_subset = noisy_labels[indices]
        
        # Count frequency of each noisy label j
        for j in range(num_classes):
            T_true[i, j] = np.sum(noisy_subset == j) / len(indices)
            
    return T_true


def evaluate_T_matrix(T_estimated, T_true):
    """
    Evaluate estimated T matrix by computing MAE and visualizing heatmaps.
    
    Args:
        T_estimated: Estimated transition matrix
        T_true: Ground truth transition matrix
    """
    # 1. Calculate MAE
    mae = np.mean(np.abs(T_true - T_estimated))
    print(f"Mean Absolute Error (MAE) between T_estimated and T_true: {mae:.4f}")
    print("   (Lower is better)")

    # 2. Plot heatmaps for comparison
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    sns.heatmap(T_estimated, annot=True, fmt='.2f', cmap='Blues', ax=axes[0])
    axes[0].set_title('Estimated T Matrix', fontsize=14)
    axes[0].set_xlabel('Noisy Label')
    axes[0].set_ylabel('Clean Label (estimated)')
    
    sns.heatmap(T_true, annot=True, fmt='.2f', cmap='Blues', ax=axes[1])
    axes[1].set_title('Ground Truth T Matrix', fontsize=14)
    axes[1].set_xlabel('Noisy Label')
    axes[1].set_ylabel('Clean Label (true)')
    
    plt.tight_layout()
    plt.show()


def estimate_T_soft(robust_model_probs, noisy_labels_one_hot, num_classes):
    """
    Estimate noise transition matrix using soft labels.
    
    T[i,j] = P(noisy=j | clean=i)
    
    Args:
        robust_model_probs: Probability output from robust model P(clean|x), shape (N, C)
        noisy_labels_one_hot: Current noisy labels as one-hot/soft labels, shape (N, C)
        num_classes: Number of classes
        
    Returns:
        Estimated transition matrix (C, C)
    """
    device = robust_model_probs.device
    
    # P(clean=i | x)
    p_clean = robust_model_probs
    # P(noisy=j | x) observed
    p_noisy_obs = noisy_labels_one_hot

    # Estimate T_ij = sum_k(P(clean=i|x_k) * P(noisy=j|x_k)) / sum_k(P(clean=i|x_k))
    # Numerator: torch.matmul(p_clean.T, p_noisy_obs) -> (C, N) @ (N, C) = (C, C)
    numerator = torch.matmul(p_clean.T, p_noisy_obs)

    # Denominator: sum_k(P(clean=i|x_k))
    denominator = torch.sum(p_clean, dim=0)

    # Normalize to get T[i, j]
    # Add epsilon to avoid division by zero
    T_estimated = numerator / (denominator.unsqueeze(1) + 1e-8)

    return T_estimated.cpu().numpy()

