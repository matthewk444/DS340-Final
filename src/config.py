"""Shared settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


def model_path(name, target="region"):
    """Checkpoint path. Region models: cnn_<name>.pt, object models: cnn_<name>_object.pt."""
    return RESULTS_DIR / (f"cnn_{name}.pt" if target == "region" else f"cnn_{name}_{target}.pt")


# Objects whose drawing style plausibly differs between countries.
CATEGORIES = [
    "house", "power outlet", "bread", "mailbox", "bus",
    "car", "tree", "cup", "chair", "face",
]

PER_REGION = 2000   # drawings kept per (object, region)
IMG_SIZE = 64
SEED = 340
