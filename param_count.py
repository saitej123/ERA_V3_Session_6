import torch
from model import Net
from torchsummary import torchsummary

model = Net()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Print model summary
torchsummary.summary(model, (1, 28, 28))

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f'\nTotal parameters: {total_params:,}') 