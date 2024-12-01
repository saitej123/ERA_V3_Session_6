import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block: maintaining spatial dimensions
        self.conv1 = nn.Conv2d(1, 10, 3, padding=1)  # 28x28x10
        self.bn1 = nn.BatchNorm2d(10)
        
        # First Block: slight channel increase
        self.conv2 = nn.Conv2d(10, 16, 3, padding=1)  # 28x28x16
        self.bn2 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)  # 14x14x16
        self.dropout1 = nn.Dropout(0.15)
        
        # Second Block: maintain channels, extract features
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1)  # 14x14x16
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 16, 3, padding=1)  # 14x14x16
        self.bn4 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2, 2)  # 7x7x16
        self.dropout2 = nn.Dropout(0.15)
        
        # Third Block: increase channels for final feature extraction
        self.conv5 = nn.Conv2d(16, 32, 3, padding=1)  # 7x7x32
        self.bn5 = nn.BatchNorm2d(32)
        self.dropout3 = nn.Dropout(0.15)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)  # 1x1x32
        
        # Final 1x1 conv instead of linear
        self.conv6 = nn.Conv2d(32, 10, 1)  # 1x1x10

    def forward(self, x):
        # Input block
        x = F.relu(self.bn1(self.conv1(x)))
        
        # First block
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second block with residual connection
        identity = x
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x = F.relu(x + identity)  # residual connection
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Third block
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.dropout3(x)
        
        # GAP and final conv
        x = self.gap(x)
        x = self.conv6(x)
        x = x.view(-1, 10)
        
        return F.log_softmax(x, dim=1) 