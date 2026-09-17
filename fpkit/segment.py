"""Foreground segmentation by local variance.

Ridges produce high local variance; background (glass, paper, blank sensor)
is flat. This fails on noisy backgrounds -- that failure is a lesson, and
it's exactly why FingerNet and friends learn segmentation instead.
"""
import cv2
import numpy as np


def segment(img: np.ndarray, block: int = 16, rel_thresh: float = 0.2) -> np.ndarray:
    norm = (img - img.mean()) / (img.std() + 1e-8)
    h, w = norm.shape
    hb, wb = h // block, w // block
    blocks = norm[: hb * block, : wb * block].reshape(hb, block, wb, block)
    std_map = blocks.std(axis=(1, 3))

    fg = (std_map > rel_thresh * std_map.max()).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)

    # keep the largest connected component (the finger)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        fg = (labels == largest).astype(np.uint8)

    mask = cv2.resize(fg, (wb * block, hb * block), interpolation=cv2.INTER_NEAREST)
    full = np.zeros((h, w), dtype=bool)
    full[: hb * block, : wb * block] = mask.astype(bool)
    return full
