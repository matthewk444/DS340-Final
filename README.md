# Doodle Region CNN

DS 340 final project (Fall 2026) — Lauren Kang, Matthew Kim.

Can a CNN guess which region of the world someone is from by looking at how they doodle?
We use the [Google Quick, Draw! dataset](https://github.com/googlecreativelab/quickdraw-dataset),
which tags every drawing with the artist's country.

## Status

- [x] **Part 1** — data: download, map countries to regions, balance, render, split
- [x] **Part 2** — train an initial CNN, compare with simple baselines
- [ ] **Part 3** — partial drawings, compare training approaches, drawing interface
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

## Data

- **10 objects:** house, power outlet, bread, mailbox, bus, car, tree, cup, chair, face.
- **9 regions** (`src/regions.py`): North America, Latin America, Western Europe, Eastern Europe,
  Middle East & Africa, South Asia, East Asia, Southeast Asia, Oceania.
- Only drawings Quick Draw recognized as the right object are used.
- **Balanced:** for every object, each region contributes the same number of drawings
  (capped by the smallest region, usually South Asia). 122,760 drawings total.
- **Split:** 80 / 10 / 10 train / val / test, stratified by object and region.

## Model

`src/model.py`: three conv blocks (conv → batch norm → ReLU → max pool; 32, 64, 128 filters),
then a 256-unit layer with dropout 0.4 and a 9-way output. Input is a 64×64 grayscale image.
Trained with Adam, cosine learning-rate schedule, 10 epochs; the checkpoint with the best
validation accuracy is kept.

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

## Code sources

- Data format and URLs: Quick, Draw! dataset documentation.
- Reference project: [google_sketcher](https://github.com/elleryqueenhomels/google_sketcher).
- Claude (Anthropic) helped write the pipeline and training code; we reviewed and
  will document our own changes here as the project continues.
