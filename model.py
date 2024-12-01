import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.conv1 = nn.Conv2d(1, 10, 3, padding=1)  # 28x28x10
        self.bn1 = nn.BatchNorm2d(10)
        
        # First Block with Residual
        self.conv2 = nn.Conv2d(10, 10, 3, padding=1, groups=2)  # 28x28x10 (grouped)
        self.bn2 = nn.BatchNorm2d(10)
        self.conv3 = nn.Conv2d(10, 16, 3, padding=1)  # 28x28x16
        self.bn3 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)  # 14x14x16
        self.dropout1 = nn.Dropout(0.15)
        
        # Second Block with Residual and Dilation
        self.conv4 = nn.Conv2d(16, 16, 3, padding=2, dilation=2, groups=2)  # 14x14x16 (grouped)
        self.bn4 = nn.BatchNorm2d(16)
        self.conv5 = nn.Conv2d(16, 16, 3, padding=1)  # 14x14x16
        self.bn5 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2, 2)  # 7x7x16
        self.dropout2 = nn.Dropout(0.15)
        
        # Third Block with Point-wise and Depth-wise Conv
        self.conv6 = nn.Conv2d(16, 16, 3, padding=1, groups=16)  # 7x7x16 (depth-wise)
        self.conv7 = nn.Conv2d(16, 10, 1)  # 7x7x10 (point-wise)
        self.bn6 = nn.BatchNorm2d(10)
        self.dropout3 = nn.Dropout(0.15)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)  # 1x1x10
        
    def forward(self, x):
        # Input block with skip connection
        x = self.bn1(F.relu(self.conv1(x)))
        
        # First block with residual
        identity1 = x
        x = self.bn2(F.relu(self.conv2(x)))  # Grouped conv
        x = self.bn3(F.relu(self.conv3(x)))
        # Skip connection after channel matching
        identity1 = F.pad(identity1, (0, 0, 0, 0, 0, 6))  # Pad channels from 10 to 16
        x = x + identity1
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second block with residual and dilation
        identity2 = x
        x = self.bn4(F.relu(self.conv4(x)))  # Grouped dilated conv
        x = self.bn5(F.relu(self.conv5(x)))
        x = x + identity2
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Third block with separable conv
        x = F.relu(self.conv6(x))  # depth-wise
        x = self.bn6(F.relu(self.conv7(x)))  # point-wise
        x = self.dropout3(x)
        
        # GAP and final output
        x = self.gap(x)
        x = x.view(-1, 10)
        
        return F.log_softmax(x, dim=1) 