# MNIST Classification Model with PyTorch

This repository contains a PyTorch implementation of a CNN model for MNIST digit classification that meets specific architectural and performance requirements.

## Model Architecture Requirements

The model is designed and tested to meet the following requirements:

1. **Parameter Count** ✓
   - Total parameters must be less than 20,000
   - Current model parameters:
     - Input Block: 1 → 8 channels
     - First Block: 8 → 8 channels (with residual)
     - Second Block: 8 → 12 channels (with residual)
     - Third Block: 12 → 10 channels (with dilation)

2. **Batch Normalization** ✓
   - Uses BatchNorm2d after each convolution layer
   - Total 6 BatchNorm layers

3. **Dropout** ✓
   - Uses Dropout(0.15) at three stages
   - After each block for regularization

4. **Global Average Pooling** ✓
   - Uses GAP instead of fully connected layers
   - Final spatial reduction to 1x1

5. **Performance Target**
   - Target: 99.4% validation accuracy
   - Current best: 98.36%
   - Training for 20 epochs

## Model Architecture Details

```python
# Input Block (28x28x1 → 28x28x8)
Conv2d(1, 8, 3, padding=1)
BatchNorm2d(8)

# First Block with Residual (28x28x8 → 14x14x8)
Conv2d(8, 8, 3, padding=1)
BatchNorm2d(8)
Conv2d(8, 8, 3, padding=1)
BatchNorm2d(8)
MaxPool2d(2, 2)
Dropout(0.15)

# Second Block with Residual (14x14x8 → 7x7x12)
Conv2d(8, 12, 3, padding=1)
BatchNorm2d(12)
Conv2d(12, 12, 3, padding=1)
BatchNorm2d(12)
MaxPool2d(2, 2)
Dropout(0.15)

# Third Block with Dilation (7x7x12 → 7x7x10)
Conv2d(12, 10, 3, padding=2, dilation=2)
BatchNorm2d(10)
Dropout(0.15)

# Output Block (7x7x10 → 10)
AdaptiveAvgPool2d(1)
```

## Training Details

1. **Optimizer & Learning Rate**
   - Optimizer: Adam with weight decay 1e-4
   - Learning Rate Schedule: OneCycleLR
     - Max LR: 0.01
     - Pct Start: 0.2
     - Div Factor: 10
     - Final Div Factor: 100

2. **Data Augmentation**
   - Random Rotation (-10°, 10°)
   - Random Affine (translate, scale, shear)
   - Random Perspective
   - Random Noise (factor=0.05)
   - Cutout (1 hole, size=8)
   - Normalization (mean=0.1307, std=0.3081)

3. **Training Configuration**
   - Batch Size: 64
   - Epochs: 20
   - Loss Function: NLL Loss
   - Early Stopping: At 99.4% accuracy

## Test Cases

```python
def test_parameter_count(model):
    # Ensures model has < 20k parameters
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 20000

def test_batch_norm_usage(model):
    # Verifies BatchNorm usage
    has_bn = any(isinstance(m, nn.BatchNorm2d) for m in model.modules())
    assert has_bn

def test_dropout_usage(model):
    # Verifies Dropout usage
    has_dropout = any(isinstance(m, nn.Dropout) for m in model.modules())
    assert has_dropout

def test_gap_usage(model):
    # Verifies Global Average Pooling usage
    has_gap = any(isinstance(m, nn.AdaptiveAvgPool2d) for m in model.modules())
    assert has_gap
```

## Current Results

- Parameters: Under 20k ✓
- Best Test Accuracy: 98.36%
- Training Time: 20 epochs
- All architectural requirements met ✓

## Running the Code

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python train.py
```

3. Run tests:
```bash
python test.py
```

## Future Improvements

1. Increase model capacity while staying under 20k params
2. Fine-tune learning rate schedule
3. Experiment with different augmentation strategies
4. Try different optimizers and regularization techniques

## License

This project is open-source and available under the MIT License.