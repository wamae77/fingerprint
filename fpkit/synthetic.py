"""Synthetic ridge patterns with KNOWN orientation and frequency.

This is your ground truth. Real fingerprints have no labels for orientation
or frequency, so the only way to know your estimators are correct (and your
angle conventions consistent) is to test on images where you know the answer.
"""
import numpy as np


def concentric_rings(size: int = 320, period: float = 9.0, noise: float = 0.0,
                     seed: int = 0):
    """Whorl-like rings. Ridge direction is tangent to the circle."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    cx = cy = size / 2
    r = np.hypot(x - cx, y - cy)
    clean = 0.5 + 0.5 * np.cos(2 * np.pi * r / period)
    theta_true = np.mod(np.arctan2(y - cy, x - cx) + np.pi / 2, np.pi)
    img = clean + noise * rng.standard_normal(clean.shape)
    return img.astype(np.float32), clean.astype(np.float32), theta_true.astype(np.float32)
