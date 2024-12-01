import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import OneCycleLR
from model import Net
from tqdm import tqdm

def train(model, device, train_loader, optimizer, scheduler, epoch):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        
        pbar.set_description(desc=f'Epoch={epoch} Loss={loss.item():.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}%')

def test(model, device, test_loader):
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
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy

def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Training hyperparameters
    batch_size = 64
    epochs = 20
    max_lr = 0.01
    
    # Data augmentation and normalization
    train_transform = transforms.Compose([
        transforms.RandomRotation((-7.0, 7.0), fill=(0,)),  # Slight rotation
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random shift
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transform)
    test_data = datasets.MNIST('../data', train=False, transform=test_transform)
    
    kwargs = {'batch_size': batch_size, 'num_workers': 2, 'pin_memory': True} if use_cuda else {'batch_size': batch_size}
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, shuffle=False, **kwargs)
    
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=max_lr/10, weight_decay=1e-4)
    
    # One Cycle Learning Rate Schedule
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.2,  # Peak at 20% of training
        div_factor=10,  # Initial lr is max_lr/10
        final_div_factor=100,  # Final lr is max_lr/100
    )
    
    best_acc = 0
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, scheduler, epoch)
        accuracy = test(model, device, test_loader)
        
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), "mnist_model.pt")
            print(f'Best accuracy: {best_acc:.2f}%')
        
        if accuracy >= 99.4:
            print(f'Target accuracy of 99.4% achieved in {epoch} epochs!')
            break

if __name__ == '__main__':
    main() 