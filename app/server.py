"""Drawing UI server.

Serves the page, predicts a region from the strokes drawn on the canvas, and
exposes the test-set metrics for the dashboard.

Run (from the repo root):  .venv/bin/python -m uvicorn app.server:app --port 8340
"""
import json
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.config import RESULTS_DIR, model_path
from src.model import MODELS
from src.regions import REGIONS
from src.render import normalize, render

MODEL_NAME = "deep"

model = MODELS[MODEL_NAME]()
model.load_state_dict(torch.load(model_path(MODEL_NAME), map_location="cpu"))
model.eval()

app = FastAPI()
HERE = Path(__file__).parent


class Drawing(BaseModel):
    strokes: list[list[list[float]]]  # [[xs, ys], ...] in canvas pixels


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


@app.get("/confusion_matrix.png")
def confusion_png():
    return FileResponse(RESULTS_DIR / "confusion_matrix.png")


@app.get("/metrics")
def metrics():
    """Test-set results, trimmed to what the dashboard shows."""
    full = json.loads((RESULTS_DIR / "metrics.json").read_text())
    return {
        "model": MODEL_NAME,
        "regions": REGIONS,
        "test_size": full["test_size"],
        "chance": full["chance"],
        "majority_class": full["majority_class"],
        "logistic_regression": full["logistic_regression"],
        "cnn": {n: {k: v[k] for k in ("accuracy", "top3", "per_region")}
                for n, v in full["cnn"].items()},
    }


@app.post("/predict")
def predict(d: Drawing):
    strokes = [s for s in d.strokes if len(s) == 2 and len(s[0]) > 0]
    if not strokes:
        raise HTTPException(400, "empty drawing")
    img = render(normalize(strokes))
    x = torch.from_numpy(img).float().div(255).view(1, 1, *img.shape)
    with torch.no_grad():
        probs = torch.softmax(model(x), 1)[0].tolist()
    return {"regions": REGIONS, "probs": probs}
