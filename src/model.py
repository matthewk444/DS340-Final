"""CNNs for the doodle images (input: 1 x 64 x 64 grayscale).

Written the same way as the CNNs in class (HW2): conv layers with 3x3 kernels and
no padding, ReLU, 2x2 max pooling, then flatten -> Linear -> dropout -> Linear.

Size bookkeeping (3x3 conv without padding: -2, 2x2 pool: // 2):

small (the MNIST network from lecture, adapted to 64x64):
    64 -conv1-> 62 -conv2-> 60 -pool-> 30      => 64 * 30 * 30 inputs to fc1

deep (conv-pool blocks, doubling the feature maps each time):
    64 -conv1-> 62 -pool-> 31
       -conv2-> 29 -pool-> 14
       -conv3-> 12 -pool-> 6
       -conv4->  4 -pool-> 2                   => 256 * 2 * 2 inputs to fc1
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .regions import REGIONS


class SmallCNN(nn.Module):
    def __init__(self, n_classes=len(REGIONS)):
        super().__init__()
        # Two conv layers: 1 input channel (grayscale) -> 32 feature maps -> 64 feature maps
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)  # halves the width and height
        self.dropout = nn.Dropout(0.5)  # randomly zeroes half the hidden units while training (less overfitting)
        # Dense layers: the 64 feature maps of size 30x30 are flattened into one long vector
        self.fc1 = nn.Linear(64 * 30 * 30, 64)
        self.fc2 = nn.Linear(64, n_classes)  # one score per class

    def forward(self, x):
        x = F.relu(self.conv1(x))                 # (B, 32, 62, 62)
        x = self.pool(F.relu(self.conv2(x)))      # (B, 64, 30, 30)
        x = torch.flatten(x, 1)                   # (B, 57600)
        x = F.relu(self.fc1(x))                   # (B, 64)
        x = self.dropout(x)
        x = self.fc2(x)                           # (B, n_classes) raw scores; CrossEntropyLoss applies softmax
        return x


class DeepCNN(nn.Module):
    def __init__(self, n_classes=len(REGIONS)):
        super().__init__()
        # Four conv layers, doubling the number of feature maps each time
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        # Pooling after every conv shrinks the image to 2x2, so the first dense layer is small
        self.fc1 = nn.Linear(256 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        # Each line is one conv -> ReLU -> pool block
        x = self.pool(F.relu(self.conv1(x)))      # (B, 32, 31, 31)
        x = self.pool(F.relu(self.conv2(x)))      # (B, 64, 14, 14)
        x = self.pool(F.relu(self.conv3(x)))      # (B, 128, 6, 6)
        x = self.pool(F.relu(self.conv4(x)))      # (B, 256, 2, 2)
        x = torch.flatten(x, 1)                   # (B, 1024)
        x = F.relu(self.fc1(x))                   # (B, 64)
        x = self.dropout(x)
        x = self.fc2(x)                           # (B, n_classes)
        return x


# Lets the command line pick a model by name: --model small / --model deep
MODELS = {"small": SmallCNN, "deep": DeepCNN}
