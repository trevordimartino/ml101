import torch
import torch.nn as nn
import torch.nn.functional as F


class DieNet(nn.Module):
    def __init__(self, n_classes=6):
        super(DieNet, self).__init__()
        self.n_classes = n_classes

        # Expected 480x480 three-channel input
        self.conv1 = nn.Conv2d(3, 9, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(9, 20, 3, dilation=2)
        self.conv3 = nn.Conv2d(20, 42, 8)
        self.conv4 = nn.Conv2d(42, 24, 4)
        self.pool4 = nn.MaxPool2d(4, 4)

        self.decision = nn.Conv2d(24, n_classes, 1)
        self.decision_pool = nn.MaxPool2d(12, 12)

    def forward(self, x):
        x = self.pool2(F.relu(self.conv1(x)))
        # Batch, C, H, W = (None, 9, 238, 238)
        x = self.pool2(F.relu(self.conv2(x)))
        # Batch, C, H, W = (None, 20, 117, 117)
        x = self.pool2(F.relu(self.conv3(x)))
        # Batch, C, H, W = (None, 42, 55, 55)
        x = self.pool4(F.relu(self.conv4(x)))
        # Batch, C, H, W = (None, 24, 12, 12)
        x = self.decision_pool(F.relu(self.decision(x)))
        # Batch, C, H, W = (None, n_classes, 1, 1)

        x = torch.squeeze(x)
        # Batch, C, ... = (None, n_classes)

        return x
