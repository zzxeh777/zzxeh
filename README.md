# Malware Detection System

A deep learning-based malware detection system for PE (Portable Executable) files. It supports both feature-based and image-based classification approaches, with an interactive web interface for real-time file scanning.

## Key Features

- **Attention-Based Feature Fusion**: Group-wise attention mechanism over 2381-dim EMBER features for adaptive feature weighting
- **Multi-Model Heterogeneous Ensemble**: CNN1D + Transformer + MLP ensemble with attention-weighted fusion
- **DL+ML Stacking Ensemble**: Deep learning base learners + XGBoost meta-classifier for hybrid decision
- **Lightweight Optimization**: Knowledge distillation, weight pruning, and INT8 quantization for edge deployment
- **Image-Based Classification**: Binary-to-image conversion with MobileNet, EfficientNet, and SimpleCNN
- **Web Application**: Flask/FastAPI-based UI for uploading and scanning PE files in real time

## Project Structure

```
malware_detection/
├── src/
│   ├── core/                          # Model definitions
│   │   ├── models.py                  # MLP, CNN1D, Attention Network
│   │   ├── heterogeneous_ensemble.py  # Heterogeneous ensemble models
│   │   ├── stacking_ensemble.py       # DL+ML Stacking ensemble
│   │   └── lightweight_optimization.py
│   ├── training/                      # Training pipeline
│   │   ├── training.py
│   │   ├── evaluation.py
│   │   ├── data_preprocessing.py
│   │   └── ember_official_extractor.py
│   └── utils/
│       ├── binary_to_image.py
│       └── visualization.py
├── app.py                             # Flask web application
├── app_fastapi.py                     # FastAPI web application
├── train_ember.py                     # Attention model training
├── train_heterogeneous_ensemble.py    # Heterogeneous ensemble training
├── train_stacking_ensemble.py         # Stacking ensemble training
├── train_lightweight.py               # Distillation & pruning
├── train_big2015.py                   # Image-based model training
├── run_ablation.py                    # Ablation experiments
└── requirements.txt
```

## Models

| Model | Type | Input | Description |
|-------|------|-------|-------------|
| Attention Network | Feature | EMBER 2381-dim | Group-wise attention with learnable weights |
| MLP | Feature | EMBER features | Multi-layer perceptron with batch normalization |
| CNN1D | Feature | EMBER features | 1D convolutional network for local pattern extraction |
| Transformer | Feature | EMBER features | Self-attention for long-range dependency |
| MobileNet | Image | 64x64 grayscale | Lightweight CNN for binary-to-image classification |
| EfficientNet | Image | 64x64 grayscale | Efficient scaling architecture |

## Quick Start

```bash
# Install dependencies
pip install -r malware_detection/requirements.txt

# Train the attention model
python malware_detection/train_ember.py

# Train heterogeneous ensemble
python malware_detection/train_heterogeneous_ensemble.py

# Run ablation experiments
python malware_detection/run_ablation.py

# Start the web application
python malware_detection/app.py
```

## Experimental Results

### Feature-Based Detection (EMBER Dataset)

| Metric | Value |
|--------|-------|
| Test Accuracy | 94.10% |
| Precision | 96.45% |
| Recall | 91.43% |
| F1 Score | 93.87% |
| AUC | 98.45% |

### Image-Based Detection (BIG2015 Dataset)

| Model | Validation Accuracy | Parameters |
|-------|-------------------|------------|
| MobileNet | 99.30% | 3.2M |
| SimpleCNN | 100.0% | - |
| EfficientNet | 96.61% | - |

### Ablation Study

| Variant | Params | Accuracy |
|---------|--------|----------|
| Full Attention (baseline) | 162K | 84.23% |
| Without Attention | 219K | 83.20% |
| Without Grouping | 643K | 83.35% |
| Fixed Weights (0.25) | 161K | 83.25% |

## Datasets

- **EMBER**: Open-source dataset for feature-based PE malware detection (2381-dim feature vectors)
- **BIG2015**: Malware classification dataset with binary-to-image conversion (9 families, 10,868 samples)

## Tech Stack

- **Framework**: PyTorch
- **Web**: Flask / FastAPI
- **PE Parsing**: pefile
- **Image Processing**: Pillow, torchvision
- **ML Meta-Learner**: XGBoost

## License

For academic research use.
