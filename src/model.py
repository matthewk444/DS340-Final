"""Small CNN: 3 conv blocks -> classifier with dropout."""
import torch.nn as nn

from .regions import REGIONS


def block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class RegionCNN(nn.Module):
    def __init__(self, n_classes=len(REGIONS), dropout=0.4):
        super().__init__()
        self.features = nn.Sequential(block(1, 32), block(32, 64), block(64, 128))  # 64 -> 8
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
