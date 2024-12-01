# MNIST Classification with PyTorch

This repository contains a PyTorch implementation of a CNN model for MNIST digit classification that achieves >99.4% test accuracy with less than 20k parameters.

## Model Architecture

The model uses:
- Batch Normalization
- Dropout (0.1)
- Global Average Pooling
- Less than 20k parameters
- Achieves >99.4% test accuracy in less than 20 epochs

## Requirements

```bash
pip install torch torchvision tqdm torchsummary
```

## Training

To train the model:

```bash
python train.py
```

## Testing

To run the model tests:

```bash
python test.py
```

## Model Verification

The GitHub Actions workflow automatically verifies:
- Parameter count is less than 20k
- Use of Batch Normalization
- Use of Dropout
- Use of Global Average Pooling

## Test Logs

The model achieves the following metrics:
- Parameters: <20k
- Test Accuracy: >99.4%
- Training Time: <20 epochs

## GitHub Actions Status
[![Model Tests](https://github.com/{username}/{repo}/actions/workflows/model_checks.yml/badge.svg)](https://github.com/{username}/{repo}/actions/workflows/model_checks.yml)

Note: Replace `{username}` and `{repo}` with your actual GitHub username and repository name. 