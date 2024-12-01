import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)  # 28x28x8
        self.bn1 = nn.BatchNorm2d(8)
        
        # First Block with Residual
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)  # 28x28x8
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 8, 3, padding=1)  # 28x28x8
        self.bn3 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(2, 2)  # 14x14x8
        self.dropout1 = nn.Dropout(0.15)
        
        # Second Block with Residual
        self.conv4 = nn.Conv2d(8, 12, 3, padding=1)  # 14x14x12
        self.bn4 = nn.BatchNorm2d(12)
        self.conv5 = nn.Conv2d(12, 12, 3, padding=1)  # 14x14x12
        self.bn5 = nn.BatchNorm2d(12)
        self.pool2 = nn.MaxPool2d(2, 2)  # 7x7x12
        self.dropout2 = nn.Dropout(0.15)
        
        # Third Block with Dilation
        self.conv6 = nn.Conv2d(12, 10, 3, padding=2, dilation=2)  # 7x7x10
        self.bn6 = nn.BatchNorm2d(10)
        self.dropout3 = nn.Dropout(0.15)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)  # 1x1x10
        
    def forward(self, x):
        # Input block
        x = self.bn1(F.relu(self.conv1(x)))
        
        # First block with residual
        identity1 = self.conv2(x)
        x = self.bn2(F.relu(identity1))
        x = self.bn3(F.relu(self.conv3(x)))
        x = x + identity1
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second block with residual
        identity2 = self.conv4(x)
        x = self.bn4(F.relu(identity2))
        x = self.bn5(F.relu(self.conv5(x)))
        x = x + identity2
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Third block with dilation
        x = self.bn6(F.relu(self.conv6(x)))
        x = self.dropout3(x)
        
        # GAP and final output
        x = self.gap(x)
        x = x.view(-1, 10)
        
        return F.log_softmax(x, dim=1) 