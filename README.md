## 📁 Project Structure

```
Label-Fixing/
├── src/
│   ├── data/
│   │   └── data_manager.py          # Data loading and processing
│   ├── models/
│   │   └── mlp.py                   # MLP architecture
│   ├── trainers/
│   │   ├── act_trainer.py           # ACT training logic
│   │   └── correction_trainer.py    # Fine-tuning with correction loss
│   ├── losses/
│   │   └── forward_correction_loss.py  # Forward correction loss
│   ├── utils/
│   │   ├── training_utils.py        # Training utilities (seed, early stopping)
│   │   └── evaluation.py            # T-matrix estimation and evaluation
│   └── pipeline/
│       └── noise_correction_pipeline.py  # Main pipeline
├── config.py                         # Configuration settings
├── train.py                          # Main training script
└── requirements.txt                  # Dependencies
```

## 🚀 Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## 🎯 Usage

### Basic Usage

```bash
python train.py \
    --ground_truth_path /path/to/ground_truth.csv \
    --features_path /path/to/features.feather
```

### Custom Parameters

```bash
python train.py \
    --ground_truth_path /path/to/data.csv \
    --features_path /path/to/features.feather \
    --batch_size 256 \
    --seed 42
```

### Edit example.py file to run 
```bash
python example.py
```

## ⚙️ Configuration

Edit `config.py` to customize hyperparameters:

```python
config = {
    'SEED': 42,
    'BATCH_SIZE': 128,
    'MODEL_DIMS': [512, 256],
    
    # Iteration parameters
    'NUM_ITERATIONS': 30,
    'MOMENTUM_ALPHA': 0.8,
    'ITERATION_PATIENCE': 3,
    
    # ACT parameters
    'ACT_EPOCHS': 150,
    'ACT_WARMUP': 20,
    'ACT_PATIENCE': 15,
    
    # Fine-tuning parameters
    'FINETUNE_EPOCHS': 100,
    'FINETUNE_LR': 1e-5,
    'FINETUNE_PATIENCE': 7
}
```
