# CANOLA: Noise-Aware Framework for Correcting Corrupted Labels

## Introduction
We introduces CANOLA, a novel framework for robust and stable corrupted label correction. Motivated by the strong generalization capability of DNNs, CANOLA rethinks the timing and reliability of the correction signal. CANOLA decouples the learning phase from the correction phase to ensure label updates are cautiously guided by a mature, converged model state.

## The Architecture
<img width="899" height="682" alt="Ảnh màn hình 2026-01-10 lúc 19 28 29" src="https://github.com/user-attachments/assets/55219a38-92bf-4afc-8f1d-95ff3ffd1c74" />

- Overview of CANOLA: An iterative pipeline alternating between Noise Transition Matrix Construction (Phase 1) via asymmetric co-training and Corrupted Label Correction (Phase 2) using noise-aware learning and loss-triggered soft relabeling.

## Quick Start 
### Download Dataset
### Prepare Environment
```bash
# Install dependencies
pip install -r requirements.txt
```

### Running 
```bash
python train.py \
    --ground_truth_path /path/to/data.csv \
    --features_path /path/to/features.feather \
    --batch_size 256 \
    --seed 42
```

## Contact us 
If you have any questions, comments, or suggestions, please do not hesitate to contact us.
- Email: 22024505@vnu.edu.vn
