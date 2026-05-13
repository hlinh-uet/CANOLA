"""
Simple example of using the Label Fixing pipeline.
"""

from src.pipeline import NoiseCorrectionPipeline
from src.utils import set_seed
from config import get_config
from sklearn.metrics import accuracy_score


def main():
    # Get default configuration
    config = get_config()
    
    # Update with your data paths
    config['GROUND_TRUTH_PATH'] = '/path/to/your/ground_truth.csv'
    config['FEATURES_PATH'] = '/path/to/your/features.feather'
    
    # Optional: Customize parameters
    # config['BATCH_SIZE'] = 256
    # config['NUM_ITERATIONS'] = 20
    # config['MODEL_DIMS'] = [1024, 512, 256]
    
    # Set random seed for reproducibility
    set_seed(config['SEED'])
    
    # Run the pipeline
    pipeline = NoiseCorrectionPipeline(config)
    corrected_labels, true_labels, noisy_labels = pipeline.run()
    
    # Evaluate results
    original_accuracy = accuracy_score(true_labels, noisy_labels)
    corrected_accuracy = accuracy_score(true_labels, corrected_labels)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Original Accuracy:  {original_accuracy*100:.2f}%")
    print(f"Corrected Accuracy: {corrected_accuracy*100:.2f}%")
    print(f"Improvement:        +{(corrected_accuracy - original_accuracy)*100:.2f}%")
    
    # Optional: Save corrected labels
    import pandas as pd
    results_df = pd.DataFrame({
        'corrected_label': corrected_labels,
        'noisy_label': noisy_labels,
        'true_label': true_labels
    })
    results_df.to_csv('corrected_labels.csv', index=False)
    print(f"\nCorrected labels saved to 'corrected_labels.csv'")


if __name__ == "__main__":
    main()

