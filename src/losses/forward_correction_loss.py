"""Forward correction loss for label noise."""

import torch
import torch.nn as nn


class ForwardCorrectionLoss(nn.Module):
    """
    Forward correction loss that applies noise transition matrix T.
    
    This loss corrects predictions by modeling the noise process:
    P(noisy_label|x) = T @ P(clean_label|x)
    """
    
    def __init__(self, T):
        """
        Initialize ForwardCorrectionLoss.
        
        Args:
            T: Noise transition matrix (num_classes, num_classes)
               T[i,j] = P(noisy=j | clean=i)
        """
        super(ForwardCorrectionLoss, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.T = T.to(device)
        # KLDivLoss expects log-probabilities as input and probabilities as target
        self.loss_fn = nn.KLDivLoss(reduction='batchmean')

    def forward(self, logits, target_soft_labels):
        """
        Compute forward correction loss.
        
        Args:
            logits: Model output logits (batch_size, num_classes)
            target_soft_labels: Target soft labels (batch_size, num_classes)
            
        Returns:
            Loss value
        """
        # 1. Compute clean label probabilities from logits
        p_clean = nn.functional.softmax(logits, dim=1)
        
        # 2. Apply transition matrix T to get noisy label probabilities
        p_noisy = torch.matmul(p_clean, self.T)
        
        # 3. Convert to log-probabilities for KLDivLoss
        log_p_noisy = torch.log(p_noisy.clamp(min=1e-7))  # Clamp to avoid log(0)
        
        # 4. Compute KL divergence loss
        return self.loss_fn(log_p_noisy, target_soft_labels)

