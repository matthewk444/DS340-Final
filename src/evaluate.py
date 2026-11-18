"""Part 2b: evaluate the CNNs against simple baselines on the test set.

Baselines:
  - majority class: always guess the most common training region
  - logistic regression on 28x28 downsampled pixels
CNNs: every trained checkpoint in results/ (small, deep).

Writes results/metrics.json, results/RESULTS.md and
results/confusion_matrix.png (for the best CNN).

Run:  python -m src.evaluate
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from .config import CATEGORIES, RESULTS_DIR, SEED, model_path
from .data import batches, load
from .model import MODELS
from .regions import REGIONS
from .train import device

LOGREG_TRAIN = 40000  # subsample for speed


def small_pixels(images, idx):
    """28x28 version of the images, flattened, for the logistic regression baseline."""
    out = []
    for start in range(0, len(idx), 2048):
        x = torch.from_numpy(images[np.sort(idx[start:start + 2048])].astype(np.float32) / 255.0)
        out.append(F.adaptive_avg_pool2d(x.unsqueeze(1), 28).flatten(1).numpy())
    return np.concatenate(out)


@torch.no_grad()
def cnn_predict(name, images, labels, idx):
    dev = device()
    model = MODELS[name]().to(dev)
    model.load_state_dict(torch.load(model_path(name), map_location=dev))
    model.eval()
    probs = []
    for x, _ in batches(images, labels, idx, 512):
        probs.append(torch.softmax(model(x.to(dev)), 1).cpu().numpy())
    return np.concatenate(probs)


def cnn_metrics(probs, y_test, obj_test):
    pred = probs.argmax(1)
    return {
        "accuracy": float((pred == y_test).mean()),
        "top3": float((np.argsort(-probs, 1)[:, :3] == y_test[:, None]).any(1).mean()),
        "per_region": {r: float((pred[y_test == i] == i).mean()) for i, r in enumerate(REGIONS)},
        "per_object": {c: float((pred[obj_test == i] == y_test[obj_test == i]).mean())
                       for i, c in enumerate(CATEGORIES)},
        "confusion_matrix": confusion_matrix(
            y_test, pred, labels=range(len(REGIONS)), normalize="true").round(4).tolist(),
    }


def plot_confusion(cm, name):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap="Greys", vmin=0, vmax=cm.max())
    ax.set_xticks(range(len(REGIONS)), REGIONS, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(REGIONS)), REGIONS, fontsize=8)
    for i in range(len(REGIONS)):
        for j in range(len(REGIONS)):
            ax.text(j, i, f"{cm[i, j]:.0%}", ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted region")
    ax.set_ylabel("True region")
    ax.set_title(f"{name} CNN confusion matrix (test set, % of each row)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)


def main():
    images, meta = load()
    y, obj = meta["region"], meta["object"]
    train, test = meta["train"], meta["test"]  # already sorted, so batch order == idx order
    y_test, obj_test = y[test], obj[test]

    majority = np.bincount(y[train]).argmax()
    acc_major = float((y_test == majority).mean())

    rng = np.random.default_rng(SEED)
    lr_idx = np.sort(rng.choice(train, min(LOGREG_TRAIN, len(train)), replace=False))
    print(f"Fitting logistic regression on {len(lr_idx)} drawings ...", flush=True)
    logreg = LogisticRegression(max_iter=300)
    logreg.fit(small_pixels(images, lr_idx), y[lr_idx])
    acc_logreg = float((logreg.predict(small_pixels(images, test)) == y_test).mean())

    cnns = {name: cnn_metrics(cnn_predict(name, images, y, test), y_test, obj_test)
            for name in MODELS if model_path(name).exists()}
    best = max(cnns, key=lambda n: cnns[n]["accuracy"])
    plot_confusion(cnns[best]["confusion_matrix"], best)

    metrics = {
        "test_size": int(len(test)),
        "chance": 1 / len(REGIONS),
        "majority_class": acc_major,
        "logistic_regression": acc_logreg,
        "cnn": cnns,
        "best_cnn": best,
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    names = list(cnns)
    header = "| " + " | ".join(f"{n} CNN" for n in names) + " |"
    rule = "|---" * len(names) + "|"
    lines = [
        "# Results (Part 2: CNNs on complete drawings)",
        "",
        f"Test set: {len(test):,} drawings not seen in training, balanced over "
        f"{len(REGIONS)} regions (chance = {1 / len(REGIONS):.1%}).",
        "",
        "| Method | Region accuracy | Top-3 |",
        "|---|---|---|",
        f"| Always guess most common region | {acc_major:.1%} | |",
        f"| Logistic regression (28x28 pixels) | {acc_logreg:.1%} | |",
        *[f"| {n} CNN | {m['accuracy']:.1%} | {m['top3']:.1%} |" for n, m in cnns.items()],
        "",
        "## CNN accuracy by true region",
        "",
        "| Region " + header,
        "|---" + rule,
        *[f"| {r} | " + " | ".join(f"{cnns[n]['per_region'][r]:.1%}" for n in names) + " |"
          for r in REGIONS],
        "",
        "## CNN accuracy by object",
        "",
        "| Object " + header,
        "|---" + rule,
        *[f"| {c} | " + " | ".join(f"{cnns[n]['per_object'][c]:.1%}" for n in names) + " |"
          for c in sorted(CATEGORIES, key=lambda c: -cnns[best]["per_object"][c])],
        "",
        f"Confusion matrix ({best} CNN): `confusion_matrix.png`. Example drawings: `samples.png`.",
    ]
    (RESULTS_DIR / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[4:9 + len(names)]))


if __name__ == "__main__":
    main()
