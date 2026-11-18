# Doodle Region CNN

DS 340 final project (Fall 2026) — Lauren Kang, Matthew Kim.

Can a CNN guess which region of the world someone is from by looking at how they doodle?
We use the [Google Quick, Draw! dataset](https://github.com/googlecreativelab/quickdraw-dataset),
which tags every drawing with the artist's country.

## Status

- [x] **Part 1** — data: download, map countries to regions, balance, render, split
- [x] **Part 2** — train an initial CNN, compare with simple baselines
- [x] Drawing UI + model dashboard (simple version)
- [ ] **Part 3** — partial drawings, compare training approaches
- [ ] Paper, lightning talk, demo

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

```bash
.venv/bin/python -m src.download   # ~2 min, streams Quick Draw files, keeps a sample
.venv/bin/python -m src.prepare    # ~1 min, balance + render 64x64 images + splits
.venv/bin/python -m src.train      # train the CNN (Apple GPU if available)
.venv/bin/python -m src.evaluate   # test accuracy vs. baselines -> results/
```

## Drawing UI

```bash
.venv/bin/python -m uvicorn app.server:app --port 8340
```

Open http://localhost:8340. Draw one of the 10 objects; after every stroke (and while
drawing) the deep CNN shows its predicted region, how confident it is, and the
probability for all 9 regions. Below that, a dashboard shows test-set accuracy for every
method, accuracy per region and the confusion matrix.

The canvas strokes go through the same preprocessing as the Quick, Draw! data
(`normalize` in `src/render.py`: shift to the top-left corner, scale to 0–255), then the
same `render` function used in training.

## Data

- **10 objects:** house, power outlet, bread, mailbox, bus, car, tree, cup, chair, face.
- **9 regions** (`src/regions.py`): North America, Latin America, Western Europe, Eastern Europe,
  Middle East & Africa, South Asia, East Asia, Southeast Asia, Oceania.
- Only drawings Quick Draw recognized as the right object are used.
- **Balanced:** for every object, each region contributes the same number of drawings
  (capped by the smallest region, usually South Asia). 122,760 drawings total.
- **Split:** 80 / 10 / 10 train / val / test, stratified by object and region.

## Model

Two CNNs in `src/model.py` (input: 64×64 grayscale image, output: 9 regions):

| Model | Layers | Filters | Params |
|---|---|---|---|
| `small` | 3 conv blocks, 1 conv each | 32, 64, 128 | 0.6M |
| `deep` | 4 conv blocks, 2 convs each after the first (7 convs) | 32, 64, 128, 256 | 2.2M |

Each block is conv → batch norm → ReLU (repeated) → max pool. Both end in a 256-unit
layer with dropout 0.4. Trained with Adam and a cosine learning-rate schedule; the
checkpoint with the best validation accuracy is kept.

```bash
.venv/bin/python -m src.train --model small --epochs 10
.venv/bin/python -m src.train --model deep --epochs 25   # ~90 s/epoch on an M-series Mac
```

## Results

See [`results/RESULTS.md`](results/RESULTS.md) for the full tables and
`results/confusion_matrix.png` for the confusion matrix.

## Files

| File | What it does |
|---|---|
| `src/config.py` | Objects, sample sizes, paths |
| `src/regions.py` | Country code → region |
| `src/download.py` | Stream NDJSON, reservoir-sample per region |
| `src/render.py` | Strokes → 64×64 image (single source of truth) |
| `src/prepare.py` | Balance, render, split, sample grid |
| `src/data.py` | Batches from the memory-mapped image file |
| `src/model.py` | CNN |
| `src/train.py` | Training loop |
| `src/evaluate.py` | Baselines, test metrics, confusion matrix |
| `app/server.py` | FastAPI: page, `/predict`, `/metrics` |
| `app/index.html` | Canvas, live prediction, dashboard (plain HTML/CSS/JS) |

## Code sources

- Data format and URLs: Quick, Draw! dataset documentation.
- Reference project: [google_sketcher](https://github.com/elleryqueenhomels/google_sketcher).
- Claude (Anthropic) helped write the pipeline and training code; we reviewed and
  will document our own changes here as the project continues.
