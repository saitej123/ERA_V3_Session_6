import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def add_noise(image, noise_factor=0.05):
    """Add random noise to image"""
    if torch.rand(1).item() > 0.5:  # 50% chance to add noise
        noise = torch.randn_like(image) * noise_factor
        return image + noise
    return image

def cutout(image, n_holes=1, length=8):
    """Apply cutout augmentation"""
    if torch.rand(1).item() > 0.5:  # 50% chance to apply cutout
        h = image.size(1)
        w = image.size(2)
        
        mask = torch.ones_like(image)
        for n in range(n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            
            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)
            
            mask[:, y1:y2, x1:x2] = 0
        
        return image * mask
    return image

def get_data_loaders(batch_size=128):
    # Basic transforms with mild augmentation
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomApply([
            transforms.RandomAffine(
                degrees=10,  # Mild rotation
                translate=(0.1, 0.1),  # Mild translation
                scale=(0.9, 1.1),  # Mild scaling
            )
        ], p=0.3)  # 30% chance of applying augmentation
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load datasets
    train_dataset = datasets.MNIST(
        './data', 
        train=True, 
        download=True,
        transform=train_transform
    )
    
    # Use moderate subset size
    indices = list(range(len(train_dataset)))
    subset_size = 30000  # Use 30k samples
    
    # Select balanced samples
    targets = train_dataset.targets.numpy()
    selected_indices = []
    samples_per_class = subset_size // 10
    
    for class_idx in range(10):
        class_indices = np.where(targets == class_idx)[0]
        selected_indices.extend(class_indices[:samples_per_class])
    
    # Shuffle the selected indices
    np.random.shuffle(selected_indices)
    train_dataset = Subset(train_dataset, selected_indices)
    
    test_dataset = datasets.MNIST(
        './data', 
        train=False,
        transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        pin_memory=True,
        num_workers=4
    )
    
    return train_loader, test_loader 