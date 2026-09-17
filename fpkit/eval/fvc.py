"""FVC evaluation protocol (FVC2000-2004).

Each DB: F fingers x I impressions, files named '<finger>_<impression>.tif'.
  Genuine:  every impression vs every other impression of the same finger,
            symmetric pairs skipped  -> F * I*(I-1)/2   (100 x 28 = 2800)
  Impostor: first impression of each finger vs first impression of every
            other finger             -> F*(F-1)/2        (4950)
Using the official protocol makes your numbers comparable to published ones.
"""
import re
from itertools import combinations
from pathlib import Path

_NAME = re.compile(r"^(\d+)_(\d+)$")


def index_db(db_dir: str | Path) -> dict[int, dict[int, Path]]:
    db: dict[int, dict[int, Path]] = {}
    for p in sorted(Path(db_dir).iterdir()):
        m = _NAME.match(p.stem)
        if m and p.suffix.lower() in {".tif", ".tiff", ".bmp", ".png"}:
            db.setdefault(int(m[1]), {})[int(m[2])] = p
    return db


def genuine_pairs(db):
    for finger in sorted(db):
        for a, b in combinations(sorted(db[finger]), 2):
            yield db[finger][a], db[finger][b]


def impostor_pairs(db):
    firsts = [db[f][min(db[f])] for f in sorted(db)]
    yield from combinations(firsts, 2)
