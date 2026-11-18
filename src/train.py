"""Part 2a: train the CNN to predict region from a complete drawing.

Keeps the checkpoint with the best validation accuracy in results/cnn.pt and
the per-epoch history in results/history.json.

Run:  python -m src.train [--epochs 10]
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn

from .config import MODEL_PATH, RESULTS_DIR, SEED
from .data import batches, load
from .model import RegionCNN


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@torch.no_grad()
def accuracy(model, images, labels, idx, dev):
    model.eval()
    correct = 0
    for x, y in batches(images, labels, idx, 512):
        correct += (model(x.to(dev)).argmax(1).cpu() == y).sum().item()
    return correct / len(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    dev = device()
    images, meta = load()
    labels = meta["region"]

    model = RegionCNN().to(dev)
    print(f"device={dev}  params={sum(p.numel() for p in model.parameters()):,}  "
          f"train={len(meta['train'])}  val={len(meta['val'])}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    history, best = [], 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total_loss, n = time.time(), 0.0, 0
        for x, y in batches(images, labels, meta["train"], args.batch_size, shuffle=True, rng=rng):
            x, y = x.to(dev), y.to(dev)
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(y)
            n += len(y)
        sched.step()

        train_loss = total_loss / n
        val_acc = accuracy(model, images, labels, meta["val"], dev)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc})
        mark = ""
        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            mark = "  (saved)"
        print(f"epoch {epoch:2d}  loss {train_loss:.4f}  val acc {val_acc:.2%}  "
              f"{time.time() - t0:.0f}s{mark}", flush=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Best val accuracy: {best:.2%}")


if __name__ == "__main__":
    main()
