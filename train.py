"""Main training script for noise correction pipeline."""

import argparse
import numpy as np
from sklearn.metrics import accuracy_score

from src.pipeline import NoiseCorrectionPipeline
from src.utils import set_seed
from config import get_config, get_small_config, get_large_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train noise correction pipeline'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='default',
        choices=['default', 'small', 'large'],
        help='Configuration preset to use'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed (overrides config)'
    )
    
    parser.add_argument(
        '--ground_truth_path',
        type=str,
        default=None,
        help='Path to ground truth CSV file (overrides config)'
    )
    
    parser.add_argument(
        '--features_path',
        type=str,
        default=None,
        help='Path to features feather file (overrides config)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    # Load configuration
    if args.config == 'small':
        config = get_small_config()
    elif args.config == 'large':
        config = get_large_config()
    else:
        config = get_config()
    
    # Override with command line arguments
    if args.seed is not None:
        config['SEED'] = args.seed
    if args.ground_truth_path is not None:
        config['GROUND_TRUTH_PATH'] = args.ground_truth_path
    if args.features_path is not None:
        config['FEATURES_PATH'] = args.features_path
    if args.batch_size is not None:
        config['BATCH_SIZE'] = args.batch_size
    
    # Set random seed
    set_seed(config['SEED'])
    
    # Initialize and run pipeline
    print("="*60)
    print("STARTING LABEL CORRECTION PIPELINE")
    print("="*60)
    
    pipeline = NoiseCorrectionPipeline(config)
    corrected_labels, true_labels, noisy_labels = pipeline.run()
    
    # Evaluate results
    print("\n" + "="*60)
    print("FINAL EVALUATION RESULTS")
    print("="*60 + "\n")
    
    # Original dataset quality
    acc_original = accuracy_score(true_labels, noisy_labels)
    noise_rate_original = 1 - acc_original
    
    # Corrected dataset quality
    acc_corrected = accuracy_score(true_labels, corrected_labels)
    noise_rate_corrected = 1 - acc_corrected
    
    print("--- Dataset Quality BEFORE and AFTER Correction ---")
    print(f"📈 ORIGINAL Dataset Accuracy: {acc_original*100:.2f}%")
    print(f"🔥 ORIGINAL Noise Rate: {noise_rate_original*100:.2f}%")
    print("-" * 40)
    print(f"📉 CORRECTED Dataset Accuracy: {acc_corrected*100:.2f}%")
    print(f"💧 REMAINING Noise Rate: {noise_rate_corrected*100:.2f}%")
    print("-" * 60)
    
    # Calculate improvement
    improvement = (acc_corrected - acc_original) * 100
    noise_reduction = (noise_rate_original - noise_rate_corrected) * 100
    
    print(f"\n✨ Accuracy Improvement: +{improvement:.2f}%")
    print(f"✨ Noise Reduction: -{noise_reduction:.2f}%")
    
    print("\n" + "="*60)
    print("COMPLETED!")
    print("="*60)


if __name__ == "__main__":
    main()

