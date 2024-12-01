import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import Net
from torchsummary import torchsummary
from data_loader import get_data_loaders
import sys
import os

def test_parameter_count(model):
    """Test if model has less than 20k parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params:,}')
    assert total_params < 20000, f'❌ Model has {total_params:,} parameters, exceeding limit of 20,000'
    print(f'✓ Parameter count test passed: {total_params:,} parameters (under 20k limit)')
    return total_params

def test_batch_norm_usage(model):
    """Test if model uses Batch Normalization"""
    bn_count = sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))
    assert bn_count >= 6, f'❌ Model should have at least 6 BatchNorm layers, found {bn_count}'
    print(f'✓ BatchNorm test passed: Found {bn_count} BatchNorm layers')
    return bn_count

def test_dropout_usage(model):
    """Test if model uses Dropout"""
    dropout_layers = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    dropout_count = len(dropout_layers)
    dropout_values = [layer.p for layer in dropout_layers]
    
    assert dropout_count >= 3, f'❌ Model should have at least 3 Dropout layers, found {dropout_count}'
    assert all(p >= 0.1 for p in dropout_values), '❌ Dropout values should be >= 0.1'
    print(f'✓ Dropout test passed: Found {dropout_count} Dropout layers with p={dropout_values[0]}')
    return dropout_count

def test_gap_usage(model):
    """Test if model uses Global Average Pooling"""
    gap_layers = [m for m in model.modules() if isinstance(m, nn.AdaptiveAvgPool2d)]
    assert len(gap_layers) > 0, '❌ Model does not use Global Average Pooling'
    assert gap_layers[0].output_size == (1, 1), '❌ GAP should reduce spatial dimensions to 1x1'
    print('✓ GAP test passed: Found correct Global Average Pooling layer')

def train_and_test(model, device, train_loader, test_loader, epochs=20):
    """Train and test the model"""
    optimizer = optim.Adam(model.parameters(), lr=0.01/10, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,
        div_factor=10,
        final_div_factor=100
    )
    
    best_acc = 0
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = nn.functional.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.4f}')
        
        # Testing
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += nn.functional.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'Epoch {epoch}:')
        print(f'  Test Loss: {test_loss:.4f}')
        print(f'  Test Accuracy: {accuracy:.2f}%')
        
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), 'best_model.pt')
            print(f'  New best accuracy: {best_acc:.2f}%')
        
        if accuracy >= 99.4:
            print(f'✓ Reached target accuracy of 99.4% in {epoch} epochs')
            return best_acc, epoch
    
    return best_acc, epochs

def main():
    try:
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Setup
        print('Starting model tests...')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Using device: {device}')
        
        # Initialize model
        model = Net().to(device)
        print('\nModel Architecture:')
        torchsummary.summary(model, (1, 28, 28))
        
        # Run architecture tests
        print('\nTesting model architecture...')
        total_params = test_parameter_count(model)
        bn_count = test_batch_norm_usage(model)
        dropout_count = test_dropout_usage(model)
        test_gap_usage(model)
        
        # Load dataset with augmentations
        print('\nLoading MNIST dataset with augmentations...')
        train_loader, test_loader = get_data_loaders(batch_size=64)
        
        # Train and test
        print('\nStarting training and testing...')
        best_acc, epochs = train_and_test(model, device, train_loader, test_loader)
        
        # Final assertions
        assert best_acc >= 99.4, f'❌ Failed to achieve 99.4% accuracy. Best accuracy: {best_acc:.2f}%'
        assert epochs <= 20, f'❌ Took {epochs} epochs, exceeding limit of 20'
        
        print('\n✓ All tests passed successfully! ✓')
        print(f'Summary:')
        print(f'- Parameters: {total_params:,} (< 20k)')
        print(f'- BatchNorm layers: {bn_count}')
        print(f'- Dropout layers: {dropout_count}')
        print(f'- Uses GAP: Yes')
        print(f'- Best accuracy: {best_acc:.2f}%')
        print(f'- Epochs taken: {epochs}')
        
    except AssertionError as e:
        print(f'\n❌ Test failed: {str(e)}')
        sys.exit(1)
    except Exception as e:
        print(f'\n❌ Unexpected error: {str(e)}')
        sys.exit(1)

if __name__ == '__main__':
    main() 