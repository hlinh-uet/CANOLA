"""Configuration for noise correction pipeline."""


def get_config():
    """
    Get default configuration for the pipeline.
    
    Returns:
        Dictionary containing all hyperparameters
    """
    config = {
        # Random seed for reproducibility
        'SEED': 42,
        
        # Data parameters
        'BATCH_SIZE': 128,
        'GROUND_TRUTH_PATH': "/kaggle/input/fashion-mnist-test/fashion_mnist.csv",
        'FEATURES_PATH': "/kaggle/input/fashion-mnist-test/fashion-mnist-test-clip-b16-noise/fashion-mnist-test_LLM.feather",
        
        # Model architecture
        'MODEL_DIMS': [512, 256],
        
        # --- Global iteration parameters ---
        'NUM_ITERATIONS': 30,  # Maximum number of correction iterations
        'MOMENTUM_ALPHA': 0.8,  # Momentum coefficient for label smoothing
        'ITERATION_PATIENCE': 3,  # Patience for early stopping between iterations
        
        # --- Co-training (ACT) parameters ---
        'ACT_EPOCHS': 150,  # Maximum epochs for ACT
        'ACT_WARMUP': 20,  # Warmup epochs (both models trained on all data)
        'ACT_PATIENCE': 15,  # Patience for early stopping in ACT
        
        # --- Fine-tuning parameters ---
        'FINETUNE_EPOCHS': 100,  # Maximum epochs for fine-tuning
        'FINETUNE_LR': 1e-5,  # Learning rate for fine-tuning
        'FINETUNE_PATIENCE': 7  # Patience for early stopping in fine-tuning
    }
    
    return config


# Example of how to create custom configs
def get_small_config():
    """Get config for smaller/faster experiments."""
    config = get_config()
    config.update({
        'NUM_ITERATIONS': 10,
        'ACT_EPOCHS': 50,
        'FINETUNE_EPOCHS': 30,
    })
    return config


def get_large_config():
    """Get config for more thorough training."""
    config = get_config()
    config.update({
        'MODEL_DIMS': [1024, 512, 256],
        'NUM_ITERATIONS': 50,
        'ACT_EPOCHS': 200,
        'FINETUNE_EPOCHS': 150,
    })
    return config

