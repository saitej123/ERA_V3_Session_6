import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import Net
from torchsummary import torchsummary
import sys
import os
from datetime import datetime

def log_step(message):
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f'[{timestamp}] {message}')

def test_parameter_count(model):
    """Test if model has less than 20k parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params:,}')
    assert total_params < 20000, f'❌ Model has {total_params:,} parameters, exceeding limit of 20,000'
    print(f'✓ Parameter count test passed: {total_params:,} parameters (under 20k limit)')
    return total_params

def test_batch_norm_usage(model):
    """Test if model uses Batch Normalization"""
    has_bn = any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())
    bn_count = sum(1 for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d))
    assert has_bn, '❌ Model does not use Batch Normalization'
    print(f'✓ BatchNorm test passed: Found {bn_count} BatchNorm layers')
    return bn_count

def test_dropout_usage(model):
    """Test if model uses Dropout"""
    has_dropout = any(isinstance(m, torch.nn.Dropout) for m in model.modules())
    dropout_count = sum(1 for m in model.modules() if isinstance(m, torch.nn.Dropout))
    assert has_dropout, '❌ Model does not use Dropout'
    print(f'✓ Dropout test passed: Found {dropout_count} Dropout layers')
    return dropout_count

def test_gap_usage(model):
    """Test if model uses Global Average Pooling"""
    has_gap = any(isinstance(m, torch.nn.AdaptiveAvgPool2d) for m in model.modules())
    assert has_gap, '❌ Model does not use Global Average Pooling'
    print('✓ GAP test passed: Found Global Average Pooling layer')

def train_and_validate(model, device, train_loader, val_loader, epochs=20):
    """Train and validate the model"""
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
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
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.4f}')
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += nn.functional.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        val_loss /= len(val_loader.dataset)
        accuracy = 100. * correct / total
        print(f'Epoch {epoch}:')
        print(f'  Validation Loss: {val_loss:.4f}')
        print(f'  Validation Accuracy: {accuracy:.2f}%')
        
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), 'best_model.pt')
            print(f'  New best accuracy: {best_acc:.2f}%')
        
        scheduler.step(accuracy)
        
        if accuracy >= 99.4:
            print(f'✓ Reached target accuracy of 99.4% in {epoch} epochs')
            return best_acc, epoch
    
    return best_acc, epochs

def main():
    try:
        # Setup
        log_step('Starting model tests...')
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        print(f'Using device: {device}')
        
        # Initialize model
        model = Net().to(device)
        
        # Run architecture tests
        log_step('Testing model architecture...')
        total_params = test_parameter_count(model)
        bn_count = test_batch_norm_usage(model)
        dropout_count = test_dropout_usage(model)
        test_gap_usage(model)
        
        # Load and prepare dataset
        log_step('Preparing dataset...')
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
        train_size = 50000
        val_size = 10000
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1000)
        
        # Train and validate
        log_step('Starting training and validation...')
        best_acc, epochs = train_and_validate(model, device, train_loader, val_loader)
        
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