"""CNNs for region classification.

small: 3 blocks of one conv each (32-64-128 filters)          ~0.6M params
deep:  4 blocks, two convs in each after the first (32-64-128-256 filters),
       more filters and convolutions, same dropout classifier  ~2.2M params
"""
import torch.nn as nn

from .regions import REGIONS


def conv(c_in, c_out):
    return [nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU()]


def block(c_in, c_out, n_convs=1):
    layers = conv(c_in, c_out)
    for _ in range(n_convs - 1):
        layers += conv(c_out, c_out)
    return nn.Sequential(*layers, nn.MaxPool2d(2))


def classifier(channels, n_classes, dropout):
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(4),
        nn.Flatten(),
        nn.Linear(channels * 4 * 4, 256),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(256, n_classes),
    )


class SmallCNN(nn.Module):
    def __init__(self, n_classes=len(REGIONS), dropout=0.4):
        super().__init__()
        self.features = nn.Sequential(block(1, 32), block(32, 64), block(64, 128))  # 64 -> 8
        self.classifier = classifier(128, n_classes, dropout)

    def forward(self, x):
        return self.classifier(self.features(x))


class DeepCNN(nn.Module):
    def __init__(self, n_classes=len(REGIONS), dropout=0.4):
        super().__init__()
        self.features = nn.Sequential(
            block(1, 32),        # 64 -> 32
            block(32, 64, 2),    # 32 -> 16
            block(64, 128, 2),   # 16 -> 8
            block(128, 256, 2),  # 8 -> 4
        )
        self.classifier = classifier(256, n_classes, dropout)

    def forward(self, x):
        return self.classifier(self.features(x))


MODELS = {"small": SmallCNN, "deep": DeepCNN}
