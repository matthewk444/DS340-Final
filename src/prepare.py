"""Part 1b: balance, render and split the dataset.

- For each object, keep the same number of drawings from every region
  (the smallest region count for that object), so no region dominates.
- Render every drawing to 64x64 once and store it in data/images.npy
  (memory-mapped during training, so it never has to fit in RAM).
- Stratified 80/10/10 train/val/test split by (object, region).
- Save a grid of example drawings per region to results/samples.png.

Run:  python -m src.prepare
"""
import gzip
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from .config import CATEGORIES, DATA_DIR, IMG_SIZE, RESULTS_DIR, SEED
from .regions import REGIONS
from .render import render


def main():
    rng = np.random.default_rng(SEED)
    drawings, regions, objects = [], [], []

    print(f"{'object':14s}" + "".join(f"{r[:9]:>10s}" for r in REGIONS) + "   kept/region")
    for obj_id, word in enumerate(CATEGORIES):
        with gzip.open(DATA_DIR / f"{word.replace(' ', '_')}.json.gz", "rt") as f:
            rows = json.load(f)["drawings"]
        by_region = [[d for d in rows if d["region"] == r] for r in range(len(REGIONS))]
        n = min(len(b) for b in by_region)
        print(f"{word:14s}" + "".join(f"{len(b):10d}" for b in by_region) + f"   {n}")
        for r, items in enumerate(by_region):
            for i in rng.choice(len(items), n, replace=False):
                drawings.append(items[i]["strokes"])
                regions.append(r)
                objects.append(obj_id)

    regions = np.array(regions, np.int64)
    objects = np.array(objects, np.int64)
    N = len(drawings)
    print(f"\nTotal: {N} drawings ({N // len(REGIONS)} per region)")

    images = np.lib.format.open_memmap(
        DATA_DIR / "images.npy", mode="w+", dtype=np.uint8, shape=(N, IMG_SIZE, IMG_SIZE)
    )
    for i, strokes in enumerate(drawings):
        images[i] = render(strokes)
        if (i + 1) % 20000 == 0:
            print(f"rendered {i + 1}/{N}", flush=True)
    images.flush()

    idx = np.arange(N)
    strata = objects * len(REGIONS) + regions
    train, rest = train_test_split(idx, test_size=0.2, stratify=strata, random_state=SEED)
    val, test = train_test_split(rest, test_size=0.5, stratify=strata[rest], random_state=SEED)
    np.savez(DATA_DIR / "meta.npz", region=regions, object=objects,
             train=np.sort(train), val=np.sort(val), test=np.sort(test))
    print(f"Split: train {len(train)}, val {len(val)}, test {len(test)}")

    # Example grid: 8 random drawings per region.
    RESULTS_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(len(REGIONS), 8, figsize=(8, len(REGIONS) * 1.1))
    for r, name in enumerate(REGIONS):
        picks = rng.choice(np.flatnonzero(regions == r), 8, replace=False)
        for c, i in enumerate(picks):
            ax = axes[r, c]
            ax.imshow(255 - images[i], cmap="gray")
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "samples.png", dpi=120)
    print("Saved results/samples.png")


if __name__ == "__main__":
    main()
