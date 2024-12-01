import torch
from model import Net
from torchsummary import torchsummary

def test_parameter_count():
    model = Net()
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params}')
    assert total_params < 20000, f"Model has {total_params} parameters, which exceeds the limit of 20,000"

def test_batch_norm_usage():
    model = Net()
    has_batch_norm = any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())
    assert has_batch_norm, "Model does not use Batch Normalization"

def test_dropout_usage():
    model = Net()
    has_dropout = any(isinstance(m, torch.nn.Dropout) for m in model.modules())
    assert has_dropout, "Model does not use Dropout"

def test_gap_usage():
    model = Net()
    has_gap = any(isinstance(m, torch.nn.AdaptiveAvgPool2d) for m in model.modules())
    assert has_gap, "Model does not use Global Average Pooling"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    
    # Print model summary
    torchsummary.summary(model, (1, 28, 28))
    
    # Run all tests
    test_parameter_count()
    test_batch_norm_usage()
    test_dropout_usage()
    test_gap_usage()
    
    print("All tests passed successfully!")

if __name__ == '__main__':
    main() 