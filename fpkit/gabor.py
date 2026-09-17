"""Contextual Gabor enhancement.

A Gabor filter is a sinusoid under a Gaussian envelope. Tuned to the local
ridge orientation and frequency, it reinforces true ridges and suppresses
noise, breaks and pores that don't fit the local pattern.

Instead of a kernel per pixel (slow), orientation is quantized into n_angles
bins and frequency into n_freqs bins; the image is filtered once per
(angle, freq) pair and each pixel takes the response from its own bin.
"""
import cv2
import numpy as np


def gabor_kernel(theta: float, freq: float, sigma_x: float = 4.0,
                 sigma_y: float = 4.0) -> np.ndarray:
    """Even-symmetric Gabor. theta = ridge direction; the sinusoid varies
    ACROSS ridges (x_perp), the envelope extends ALONG them (y_along)."""
    half = int(np.ceil(3 * max(sigma_x, sigma_y)))
    y, x = np.mgrid[-half: half + 1, -half: half + 1].astype(np.float32)
    x_perp = -x * np.sin(theta) + y * np.cos(theta)
    y_along = x * np.cos(theta) + y * np.sin(theta)
    envelope = np.exp(-0.5 * (x_perp ** 2 / sigma_x ** 2 + y_along ** 2 / sigma_y ** 2))
    k = envelope * np.cos(2 * np.pi * freq * x_perp)
    return (k - k.mean()).astype(np.float32)   # zero DC: flat regions -> 0


def enhance(img: np.ndarray, theta: np.ndarray, freq: np.ndarray,
            mask: np.ndarray | None = None, n_angles: int = 16,
            n_freqs: int = 4) -> np.ndarray:
    norm = (img - img.mean()) / (img.std() + 1e-8)

    f_lo, f_hi = float(np.percentile(freq, 5)), float(np.percentile(freq, 95))
    f_bins = np.linspace(f_lo, f_hi, n_freqs) if f_hi - f_lo > 1e-4 else np.array([f_lo])
    a_idx = np.round(theta / np.pi * n_angles).astype(int) % n_angles
    f_idx = np.abs(freq[..., None] - f_bins).argmin(axis=-1)

    out = np.zeros_like(norm)
    for fi, f in enumerate(f_bins):
        for ai in range(n_angles):
            sel = (a_idx == ai) & (f_idx == fi)
            if not sel.any():
                continue
            k = gabor_kernel(ai * np.pi / n_angles, float(f))
            out[sel] = cv2.filter2D(norm, cv2.CV_32F, k)[sel]

    if mask is not None:
        out[~mask] = 0
    return out
