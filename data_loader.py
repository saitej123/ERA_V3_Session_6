import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def add_noise(image, noise_factor=0.05):
    """Add random noise to image"""
    if torch.rand(1).item() > 0.5:  # 50% chance to add noise
        noise = torch.randn_like(image) * noise_factor
        return torch.clamp(image + noise, 0., 1.)
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

class MixUp:
    """Perform mixup on images and targets"""
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, batch):
        images, targets = batch
        
        # Generate mixup weights
        weights = torch.distributions.Beta(self.alpha, self.alpha).sample(torch.Size([len(images)]))
        weights = weights.to(images.device)
        
        # Create shuffled indices
        indices = torch.randperm(len(images))
        
        # Mix the images
        mixed_images = (weights.view(-1, 1, 1, 1) * images + 
                       (1 - weights.view(-1, 1, 1, 1)) * images[indices])
        
        # Mix the targets (using one-hot encoding)
        targets_onehot = torch.zeros(len(targets), 10, device=targets.device)
        targets_onehot.scatter_(1, targets.view(-1, 1), 1)
        targets_shuffled = targets_onehot[indices]
        mixed_targets = (weights.view(-1, 1) * targets_onehot + 
                        (1 - weights.view(-1, 1)) * targets_shuffled)
        
        return mixed_images, mixed_targets

def get_data_loaders(batch_size=64):
    # Advanced augmentation pipeline
    train_transform = transforms.Compose([
        transforms.RandomApply([
            transforms.RandomAffine(
                degrees=(-10, 10),
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=(-5, 5),
                fill=0
            )
        ], p=0.7),
        transforms.RandomApply([
            transforms.RandomPerspective(
                distortion_scale=0.2,
                p=0.5,
                fill=0
            )
        ], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: add_noise(x, noise_factor=0.05)),
        transforms.Lambda(lambda x: cutout(x, n_holes=1, length=8))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load datasets with advanced augmentation
    train_dataset = datasets.MNIST(
        './data', 
        train=True, 
        download=True,
        transform=train_transform
    )
    
    # Use full training set
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    test_dataset = datasets.MNIST(
        './data',
        train=False,
        transform=test_transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, test_loader 