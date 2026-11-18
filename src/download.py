"""Part 1a: download a region-balanced sample of Quick, Draw! drawings.

Streams each object's simplified NDJSON file (never stored in full) and keeps
a random sample of up to PER_REGION recognized drawings per region
(reservoir sampling). Saves data/<object>.json.gz.

Run:  python -m src.download
"""
import gzip
import json
import random
import urllib.parse
import urllib.request
from multiprocessing import Pool

from .config import CATEGORIES, DATA_DIR, PER_REGION, SEED
from .regions import REGIONS, region_of

URL = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/{}.ndjson"


def fetch(word):
    out = DATA_DIR / f"{word.replace(' ', '_')}.json.gz"
    if out.exists():
        return word, "already downloaded"

    rng = random.Random(f"{SEED}-{word}")
    kept = [[] for _ in REGIONS]
    seen = [0] * len(REGIONS)
    with urllib.request.urlopen(URL.format(urllib.parse.quote(word)), timeout=120) as resp:
        for line in resp:
            rec = json.loads(line)
            r = region_of(rec["countrycode"])
            if r is None or not rec["recognized"]:
                continue
            seen[r] += 1
            item = {"country": rec["countrycode"], "region": r, "strokes": rec["drawing"]}
            if len(kept[r]) < PER_REGION:
                kept[r].append(item)
            else:
                j = rng.randrange(seen[r])
                if j < PER_REGION:
                    kept[r][j] = item

    rows = [item for region_items in kept for item in region_items]
    with gzip.open(out, "wt") as f:
        json.dump({"object": word, "available": seen, "drawings": rows}, f)
    return word, " ".join(f"{n[:8]}={s}" for n, s in zip(REGIONS, seen))


def main():
    DATA_DIR.mkdir(exist_ok=True)
    with Pool(4) as pool:
        for word, msg in pool.imap_unordered(fetch, CATEGORIES):
            print(f"{word}: {msg}", flush=True)


if __name__ == "__main__":
    main()
