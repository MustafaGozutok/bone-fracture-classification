# Bone Fracture Classification

This project compares classical machine learning and deep learning approaches for binary bone fracture classification using the FracAtlas dataset.

The repository includes a full experimental pipeline with:

- Decision Tree
- SVM
- Naive Bayes
- K-Nearest Neighbors
- XGBoost
- CNN with EfficientNet-B0 transfer learning

## Overview

The classical machine learning pipeline uses handcrafted image descriptors and dimensionality reduction:

- HOG for shape and edge information
- LBP for local texture patterns
- GLCM for texture statistics
- StandardScaler + PCA before model training

The deep learning pipeline uses:

- EfficientNet-B0 pretrained on ImageNet
- CLAHE preprocessing
- Two-stage fine-tuning
- Grad-CAM based interpretability

## Project Structure

```text
.
|-- augment_fractured.py
|-- config.py
|-- data_loader.py
|-- evaluate.py
|-- feature_extraction.py
|-- run_all_experiments.py
|-- train_cnn.py
|-- train_decision_tree.py
|-- train_knn.py
|-- train_naive_bayes.py
|-- train_svm.py
|-- train_xgboost.py
|-- data/                # not included in the repository
|-- results/             # generated outputs
`-- saved_models/        # generated model files
```

## Dataset

This repository does not include the dataset because of size constraints.

Expected dataset location:

```text
data/FracAtlas/
```

Configured paths are defined in `config.py`.

## Installation

Create a Python environment and install the main dependencies:

```bash
pip install numpy opencv-python scikit-image scikit-learn matplotlib seaborn joblib xgboost torch torchvision
```

Optional:

- `train_svm.py` can try GPU-based SVM with RAPIDS `cuML` if it is available in the environment.
- `train_cnn.py` uses CUDA automatically when PyTorch detects a supported GPU.

## Running Experiments

Run the full benchmark:

```bash
python run_all_experiments.py
```

Run individual models:

```bash
python train_decision_tree.py
python train_svm.py
python train_naive_bayes.py
python train_knn.py
python train_xgboost.py
python train_cnn.py
```

## Outputs

Generated artifacts are saved under:

- `results/` for plots, evaluation outputs, and serialized result summaries
- `saved_models/` for trained model files

Typical outputs include:

- confusion matrices
- ROC curves
- model comparison plots
- feature importance plots
- serialized models
