"""Load the prepared dataset as tensors and DataLoaders (same steps as in class).

images.npy holds every drawing as a 64x64 uint8 image; meta.npz holds the labels
("region", "object") and the row numbers of the "train", "val" and "test" splits.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import DATA_DIR


def load():
    """All images (memory-mapped, so they are only read from disk when used) and the labels/splits."""
    images = np.load(DATA_DIR / "images.npy", mmap_mode="r")
    meta = dict(np.load(DATA_DIR / "meta.npz"))
    return images, meta


def to_tensors(images, labels, idx):
    """Images of one split as a float tensor (N, 1, 64, 64) and labels as a long tensor."""
    # unsqueeze(1) adds the channel dimension: (N, 64, 64) -> (N, 1, 64, 64)
    x = torch.tensor(images[idx], dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(labels[idx], dtype=torch.long)
    return x, y


def get_loaders(target="region", batch_size=64):
    """Train / val / test DataLoaders, normalized with the training mean and std.

    target: "region" (where the drawer is from) or "object" (what was drawn).
    Returns (train_loader, val_loader, test_loader, mean, std). Keep mean and std:
    any new drawing has to be normalized the same way before the model sees it.
    """
    images, meta = load()
    labels = meta[target]

    X_train, y_train = to_tensors(images, labels, meta["train"])
    X_val, y_val = to_tensors(images, labels, meta["val"])
    X_test, y_test = to_tensors(images, labels, meta["test"])

    # Normalize: subtract the mean, divide by the std. Both come from the training
    # data only, so nothing about the val/test sets leaks into training.
    mean, std = X_train.mean().item(), X_train.std().item()
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    # Only the training data is shuffled (a new order every epoch)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, mean, std
