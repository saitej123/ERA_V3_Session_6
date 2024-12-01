import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)  # input: 28x28x1 output: 28x28x8 RF: 3x3
        self.bn1 = nn.BatchNorm2d(8)
        self.dropout1 = nn.Dropout(0.1)

        # CONV Block 1
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)  # input: 28x28x8 output: 28x28x16 RF: 5x5
        self.bn2 = nn.BatchNorm2d(16)
        self.dropout2 = nn.Dropout(0.1)

        # Transition Block 1
        self.pool1 = nn.MaxPool2d(2, 2)  # output: 14x14x16 RF: 6x6
        self.conv3 = nn.Conv2d(16, 8, 1)  # output: 14x14x8 RF: 6x6
        self.bn3 = nn.BatchNorm2d(8)

        # CONV Block 2
        self.conv4 = nn.Conv2d(8, 16, 3, padding=1)  # output: 14x14x16 RF: 10x10
        self.bn4 = nn.BatchNorm2d(16)
        self.dropout3 = nn.Dropout(0.1)

        # Transition Block 2
        self.pool2 = nn.MaxPool2d(2, 2)  # output: 7x7x16 RF: 12x12
        self.conv5 = nn.Conv2d(16, 8, 1)  # output: 7x7x8 RF: 12x12
        self.bn5 = nn.BatchNorm2d(8)

        # Output Block
        self.conv6 = nn.Conv2d(8, 16, 3)  # output: 5x5x16 RF: 20x20
        self.bn6 = nn.BatchNorm2d(16)
        self.dropout4 = nn.Dropout(0.1)
        
        # GAP Layer
        self.gap = nn.AdaptiveAvgPool2d(1)  # output: 1x1x16
        
        # Final Conv to get 10 outputs
        self.conv7 = nn.Conv2d(16, 10, 1)  # output: 1x1x10

    def forward(self, x):
        x = self.dropout1(self.bn1(F.relu(self.conv1(x))))
        x = self.dropout2(self.bn2(F.relu(self.conv2(x))))
        x = self.bn3(F.relu(self.conv3(self.pool1(x))))
        x = self.dropout3(self.bn4(F.relu(self.conv4(x))))
        x = self.bn5(F.relu(self.conv5(self.pool2(x))))
        x = self.dropout4(self.bn6(F.relu(self.conv6(x))))
        x = self.gap(x)
        x = self.conv7(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1) 