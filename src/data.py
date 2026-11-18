"""Load the prepared dataset and serve mini-batches."""
import numpy as np
import torch

from .config import DATA_DIR


def load():
    images = np.load(DATA_DIR / "images.npy", mmap_mode="r")
    meta = dict(np.load(DATA_DIR / "meta.npz"))
    return images, meta


def batches(images, labels, idx, batch_size, shuffle=False, rng=None):
    """Yield (x, y) tensors. x: float (B, 1, 64, 64) in 0..1."""
    if shuffle:
        idx = rng.permutation(idx)
    for start in range(0, len(idx), batch_size):
        b = np.sort(idx[start:start + batch_size])  # sorted reads are faster on a memmap
        x = torch.from_numpy(images[b].astype(np.float32) / 255.0).unsqueeze(1)
        y = torch.from_numpy(labels[b])
        yield x, y
