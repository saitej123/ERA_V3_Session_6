import torch
import torch.nn as nn
from torchvision import datasets, transforms
from model import Net
from torchsummary import torchsummary
import sys
import os
from tqdm import tqdm
import requests

def download_mnist():
    """Download MNIST dataset manually if not present"""
    print('\nChecking MNIST dataset...')
    data_dir = '../data/MNIST/raw'
    processed_dir = '../data/MNIST/processed'
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # URLs for MNIST dataset files
    base_url = 'https://ossci-datasets.s3.amazonaws.com/mnist'
    files = {
        'train-images-idx3-ubyte.gz': 'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz': 'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz': 't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz': 't10k-labels-idx1-ubyte.gz'
    }

    # Download files if they don't exist
    for filename, remote_file in files.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            url = f'{base_url}/{remote_file}'
            print(f'Downloading {filename}...')
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                
                with open(filepath, 'wb') as f, tqdm(
                    desc=filename,
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:
                    for data in response.iter_content(chunk_size=1024):
                        size = f.write(data)
                        pbar.update(size)
                print(f'Successfully downloaded {filename}')
            except Exception as e:
                print(f'Error downloading {filename}: {str(e)}')
                if os.path.exists(filepath):
                    os.remove(filepath)
                return False
    return True

def test_parameter_count(model):
    """Test if model has less than 20k parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    print(f'\nTest 1: Parameter Count Check')
    print(f'Total parameters: {total_params:,}')
    assert total_params < 20000, f'❌ Model has {total_params:,} parameters, exceeding limit of 20,000'
    print(f'✓ Model has {total_params:,} parameters (less than 20k)')
    return total_params

def test_batch_norm_usage(model):
    """Test if model uses Batch Normalization"""
    print(f'\nTest 2: Batch Normalization Check')
    has_bn = any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())
    bn_count = sum(1 for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d))
    assert has_bn, '❌ Model does not use Batch Normalization'
    print(f'✓ Found {bn_count} BatchNorm layers')
    return bn_count

def test_dropout_usage(model):
    """Test if model uses Dropout"""
    print(f'\nTest 3: Dropout Check')
    has_dropout = any(isinstance(m, torch.nn.Dropout) for m in model.modules())
    dropout_count = sum(1 for m in model.modules() if isinstance(m, torch.nn.Dropout))
    assert has_dropout, '❌ Model does not use Dropout'
    print(f'✓ Found {dropout_count} Dropout layers')
    return dropout_count

def test_gap_usage(model):
    """Test if model uses Global Average Pooling"""
    print(f'\nTest 4: Global Average Pooling Check')
    has_gap = any(isinstance(m, torch.nn.AdaptiveAvgPool2d) for m in model.modules())
    gap_count = sum(1 for m in model.modules() if isinstance(m, torch.nn.AdaptiveAvgPool2d))
    assert has_gap, '❌ Model does not use Global Average Pooling'
    print(f'✓ Found {gap_count} Global Average Pooling layer(s)')
    return gap_count

def test_accuracy(model, device):
    """Test model accuracy on validation split"""
    print(f'\nTest 5: Model Accuracy Check')
    
    # First ensure dataset is downloaded
    if not download_mnist():
        print('❌ Failed to download MNIST dataset')
        return False
    
    # Data loading
    try:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Load full training dataset
        print('Loading MNIST dataset...')
        train_dataset = datasets.MNIST('../data', train=True, download=False, transform=transform)
        
        # Split into train and validation (50k/10k)
        train_size = 50000
        val_size = 10000
        _, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1000)
        print('Dataset loaded successfully')
        
    except Exception as e:
        print(f'❌ Failed to load dataset: {str(e)}')
        return False
    
    model.eval()
    correct = 0
    total = 0
    
    try:
        print('Running validation...')
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        accuracy = 100. * correct / total
        assert accuracy >= 99.4, f'❌ Model accuracy {accuracy:.2f}% is below required 99.4%'
        print(f'✓ Model achieved {accuracy:.2f}% accuracy on validation set')
        return True
        
    except Exception as e:
        print(f'❌ Error during accuracy testing: {str(e)}')
        return False

def print_model_summary(model):
    """Print model architecture summary"""
    print('\nModel Architecture Summary:')
    print('-' * 80)
    try:
        torchsummary.summary(model, (1, 28, 28))
    except Exception as e:
        print(f'Warning: Could not print model summary: {str(e)}')

def main():
    try:
        # Setup
        print('Starting model tests...')
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        print(f'Using device: {device}')
        
        # Initialize model
        model = Net().to(device)
        
        # Load best model if available
        model_path = 'best_model.pt'
        if os.path.exists(model_path):
            print(f'Loading saved model from {model_path}')
            model.load_state_dict(torch.load(model_path))
        
        # Run all tests
        print_model_summary(model)
        test_parameter_count(model)
        test_batch_norm_usage(model)
        test_dropout_usage(model)
        test_gap_usage(model)
        test_accuracy(model, device)
        
        print('\n✓ All tests passed successfully! ✓')
        
    except AssertionError as e:
        print(f'\n❌ Test failed: {str(e)}')
        sys.exit(1)
    except Exception as e:
        print(f'\n❌ Unexpected error: {str(e)}')
        sys.exit(1)

if __name__ == '__main__':
    main() 