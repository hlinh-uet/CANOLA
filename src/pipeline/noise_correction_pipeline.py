"""Main noise correction pipeline."""

import numpy as np
import torch
import torch.nn.functional as F

from ..data import DataManager
from ..trainers import ACTTrainer, CorrectionTrainer
from ..losses import ForwardCorrectionLoss
from ..utils import EarlyStopper, calculate_ground_truth_T, evaluate_T_matrix, estimate_T_soft


class NoiseCorrectionPipeline:
    """
    Complete pipeline for iterative noise correction in labels.
    
    The pipeline works in iterations, each containing:
    1. ACT training to get robust model
    2. Estimation of noise transition matrix T
    3. Fine-tuning with forward correction loss
    4. Label correction using momentum
    """
    
    def __init__(self, config):
        """
        Initialize pipeline.
        
        Args:
            config: Configuration dictionary containing all hyperparameters
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("--- Step 0: Load & Process Data ---\n")
        self.data_manager = DataManager(
            ground_truth_path=config['GROUND_TRUTH_PATH'],
            features_path=config['FEATURES_PATH'],
            batch_size=config['BATCH_SIZE']
        )
        
        self.input_dim = self.data_manager.embeddings.shape[1]
        self.num_classes = self.data_manager.num_classes
        self.noisy_labels_initial = self.data_manager.noisy_labels.copy()

    def _run_single_iteration(self):
        """
        Run a single iteration of the correction pipeline.
        
        Returns:
            Tuple of (corrected_soft_labels, final_loss)
        """
        # --- Phase 1: ACT Training and T Matrix Estimation ---
        print("\n--- Phase 1: Training ACT and Estimating T Matrix ---")
        act_trainer = ACTTrainer(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            hidden_dims=self.config['MODEL_DIMS']
        )
        act_stopper = EarlyStopper(patience=self.config['ACT_PATIENCE'])
        
        robust_model = act_trainer.train(
            self.data_manager.get_full_dataloader(shuffle=True),
            epochs=self.config['ACT_EPOCHS'],
            warmup_epochs=self.config['ACT_WARMUP'],
            early_stopper=act_stopper
        )
        
        # Estimate transition matrix
        robust_model.eval()
        all_features = self.data_manager.get_full_dataset().tensors[0].to(self.device)
        current_noisy_soft_labels = self.data_manager.get_full_dataset().tensors[1].to(self.device)
        
        with torch.no_grad():
            proxy_clean_probs = F.softmax(robust_model(all_features), dim=1)
        
        T_estimated = estimate_T_soft(
            proxy_clean_probs, 
            current_noisy_soft_labels, 
            self.num_classes
        )
        
        # Evaluate T matrix (for observation only)
        print("\n--- Evaluating T Matrix (for observation) ---")
        current_noisy_hard_labels = self.data_manager.noisy_labels
        T_true = calculate_ground_truth_T(
            self.data_manager.true_labels, 
            current_noisy_hard_labels, 
            self.num_classes
        )
        evaluate_T_matrix(T_estimated, T_true)
        
        # --- Phase 2: Fine-tuning ---
        print("\n--- Phase 2: Fine-tuning Model ---")
        final_classifier, final_loss = self._finetune_with_correction(
            T_estimated, 
            robust_model
        )
        
        print("\n--- Getting corrected soft labels from current iteration ---")
        corrected_soft_labels = self._predict_soft(final_classifier)
        
        return corrected_soft_labels, final_loss

    def _finetune_with_correction(self, T_estimated, model_to_finetune):
        """
        Fine-tune model with forward correction loss.
        
        Args:
            T_estimated: Estimated transition matrix
            model_to_finetune: Model to fine-tune
            
        Returns:
            Tuple of (trained_model, final_average_loss)
        """
        optimizer = torch.optim.Adam(
            model_to_finetune.parameters(), 
            lr=self.config['FINETUNE_LR']
        )
        correction_loss_fn = ForwardCorrectionLoss(
            T=torch.tensor(T_estimated, dtype=torch.float32)
        )
        
        correction_trainer = CorrectionTrainer(
            model_to_finetune, 
            optimizer, 
            correction_loss_fn, 
            self.device
        )
        finetune_stopper = EarlyStopper(patience=self.config['FINETUNE_PATIENCE'])
        
        trained_classifier, final_avg_loss = correction_trainer.train(
            self.data_manager.get_full_dataloader(shuffle=True),
            epochs=self.config['FINETUNE_EPOCHS'],
            early_stopper=finetune_stopper
        )
        return trained_classifier, final_avg_loss

    def _predict_soft(self, model):
        """
        Get soft label predictions from model.
        
        Args:
            model: Trained model
            
        Returns:
            Soft label predictions as numpy array
        """
        model.eval()
        all_features = self.data_manager.get_full_dataset().tensors[0].to(self.device)
        with torch.no_grad():
            final_logits = model(all_features)
            corrected_probs = F.softmax(final_logits, dim=1).cpu().numpy()
        return corrected_probs

    def run(self):
        """
        Execute complete pipeline with early stopping based on loss.
        
        Returns:
            Tuple of (corrected_labels, true_labels, initial_noisy_labels)
        """
        # Initialize early stopping mechanism
        best_loss_so_far = float('inf')
        patience_counter = 0
        best_labels_so_far = None

        current_soft_labels = self.data_manager.y_noisy_tensor.cpu().numpy()

        for i in range(self.config['NUM_ITERATIONS']):
            print("\n" + "="*60)
            print(f"🚀 STARTING CORRECTION ITERATION {i+1}/{self.config['NUM_ITERATIONS']}")
            print("="*60)

            # Run iteration
            newly_corrected_soft_labels, iteration_loss = self._run_single_iteration()
            
            print(f"\n📊 End of iteration {i+1} evaluation:")
            print(f"   - Overall Correction Loss: {iteration_loss:.4f}")

            # Early stopping logic based on loss
            if iteration_loss < best_loss_so_far:
                best_loss_so_far = iteration_loss
                # Apply momentum to compute labels to save
                alpha = self.config['MOMENTUM_ALPHA']
                updated_soft_labels_for_saving = (
                    alpha * newly_corrected_soft_labels + 
                    (1 - alpha) * current_soft_labels
                )
                best_labels_so_far = np.argmax(
                    updated_soft_labels_for_saving, 
                    axis=1
                ).copy()
                patience_counter = 0
                print(f"🎉 New improvement! Best loss so far: {best_loss_so_far:.4f}")
            else:
                patience_counter += 1
                print(f"📉 No loss improvement. Patience: {patience_counter}/{self.config['ITERATION_PATIENCE']}")

            if patience_counter >= self.config['ITERATION_PATIENCE']:
                print(
                    f"\n🛑 Early stopping: loss has not improved for "
                    f"{self.config['ITERATION_PATIENCE']} iterations."
                )
                break

            # Update soft labels for next iteration
            alpha = self.config['MOMENTUM_ALPHA']
            updated_soft_labels = (
                alpha * newly_corrected_soft_labels + 
                (1 - alpha) * current_soft_labels
            )
            self.data_manager.update_noisy_soft_labels(updated_soft_labels)
            current_soft_labels = updated_soft_labels

        print("\n" + "="*60)
        print("🎉 ITERATIVE LEARNING PROCESS COMPLETED! 🎉")
        print(f"🏆 Lowest correction loss achieved: {best_loss_so_far:.4f}")
        print("="*60)

        # Return best labels found
        final_labels = (
            best_labels_so_far if best_labels_so_far is not None 
            else np.argmax(current_soft_labels, axis=1)
        )
        return final_labels, self.data_manager.true_labels, self.noisy_labels_initial

