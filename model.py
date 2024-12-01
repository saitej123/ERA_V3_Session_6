import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.conv1 = nn.Conv2d(1, 12, 3, padding=1)  # 28x28x12
        self.bn1 = nn.BatchNorm2d(12)
        
        # First Block with Residual
        self.conv2 = nn.Conv2d(12, 16, 3, padding=1)  # 28x28x16
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1, groups=4)  # 28x28x16 (grouped)
        self.bn3 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)  # 14x14x16
        self.dropout1 = nn.Dropout(0.1)
        
        # Second Block with Dilation and Channel Expansion
        self.conv4 = nn.Conv2d(16, 20, 3, padding=2, dilation=2)  # 14x14x20
        self.bn4 = nn.BatchNorm2d(20)
        self.conv5 = nn.Conv2d(20, 20, 3, padding=1, groups=4)  # 14x14x20 (grouped)
        self.bn5 = nn.BatchNorm2d(20)
        self.pool2 = nn.MaxPool2d(2, 2)  # 7x7x20
        self.dropout2 = nn.Dropout(0.1)
        
        # Third Block - Efficient Feature Refinement
        self.conv6 = nn.Conv2d(20, 16, 1)  # 7x7x16 (pointwise reduction)
        self.bn6 = nn.BatchNorm2d(16)
        self.conv7 = nn.Conv2d(16, 16, 3, padding=1, groups=4)  # 7x7x16 (grouped)
        self.bn7 = nn.BatchNorm2d(16)
        self.dropout3 = nn.Dropout(0.1)
        
        # Final Block - Classification
        self.conv8 = nn.Conv2d(16, 10, 1)  # 7x7x10 (pointwise)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # 1x1x10
        
    def forward(self, x):
        # Input block
        x = F.relu(self.bn1(self.conv1(x)))
        
        # First block with residual
        identity1 = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # Skip connection with channel adjustment
        identity1 = F.pad(identity1, (0, 0, 0, 0, 0, 4))  # Pad 12->16 channels
        x = x + identity1
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second block with dilation
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Third block - feature refinement
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = self.dropout3(x)
        
        # Final classification
        x = F.relu(self.conv8(x))
        x = self.gap(x)
        x = x.view(-1, 10)
        
        return F.log_softmax(x, dim=1) 