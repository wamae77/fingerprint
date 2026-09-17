"""Image loading. Everything downstream works on float32 images in [0, 1]."""
from pathlib import Path
import cv2
import numpy as np


def load_gray(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img.astype(np.float32) / 255.0
