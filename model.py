import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)  # 28x28x8
        self.bn1 = nn.BatchNorm2d(8)
        
        # First Block
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)  # 28x28x16
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1)  # 28x28x16
        self.bn3 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)  # 14x14x16
        self.dropout1 = nn.Dropout(0.1)

        # Second Block
        self.conv4 = nn.Conv2d(16, 32, 3, padding=1)  # 14x14x32
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 32, 3, padding=1)  # 14x14x32
        self.bn5 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)  # 7x7x32
        self.dropout2 = nn.Dropout(0.1)
        
        # Third Block with Dilation
        self.conv6 = nn.Conv2d(32, 16, 3, padding=2, dilation=2)  # 7x7x16
        self.bn6 = nn.BatchNorm2d(16)
        self.dropout3 = nn.Dropout(0.1)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)  # 1x1x16
        
        # Final 1x1 conv
        self.conv7 = nn.Conv2d(16, 10, 1)  # 1x1x10

    def forward(self, x):
        # Input block
        x = self.bn1(F.relu(self.conv1(x)))
        
        # First block with residual
        identity1 = self.conv2(x)  # Match channels first
        x = self.bn2(F.relu(identity1))
        x = self.bn3(F.relu(self.conv3(x)))
        x = x + identity1  # Now dimensions match
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second block with residual
        identity2 = self.conv4(x)  # Match channels first
        x = self.bn4(F.relu(identity2))
        x = self.bn5(F.relu(self.conv5(x)))
        x = x + identity2  # Now dimensions match
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Third block with dilation for larger receptive field
        x = self.bn6(F.relu(self.conv6(x)))
        x = self.dropout3(x)
        
        # GAP and final conv
        x = self.gap(x)
        x = self.conv7(x)
        x = x.view(-1, 10)
        
        return F.log_softmax(x, dim=1) 