"""Ridge frequency estimation (oriented x-signature, Hong et al. 1998).

For each block, sample a rectangle whose long axis is perpendicular to the
ridges, average along the ridge direction, and measure peak spacing in the
resulting 1D profile. At 500 ppi, ridge period is typically ~8-12 px.
"""
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

MIN_PERIOD, MAX_PERIOD = 3.0, 25.0


def _block_period(img, cx, cy, theta, length, width):
    t = np.array([np.cos(theta), np.sin(theta)])      # along ridges
    n = np.array([-np.sin(theta), np.cos(theta)])     # across ridges
    u = np.arange(length) - length / 2
    v = np.arange(width) - width / 2
    uu, vv = np.meshgrid(u, v)                        # (width, length)
    xs = (cx + uu * n[0] + vv * t[0]).astype(np.float32)
    ys = (cy + uu * n[1] + vv * t[1]).astype(np.float32)
    patch = cv2.remap(img, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    signal = patch.mean(axis=0)                       # the x-signature
    # Smooth before peak picking. Without this, noise creates spurious peaks
    # and the period estimate collapses toward MIN_PERIOD (measured: a true
    # 9px period read as 5.8px at noise=1.0). The prominence gate does the same
    # job for small bumps that survive smoothing.
    signal = gaussian_filter1d(signal - signal.mean(), 1.0)
    peaks, _ = find_peaks(signal, distance=MIN_PERIOD, prominence=0.3 * signal.std())
    if len(peaks) < 2:
        return np.nan
    period = float(np.diff(peaks).mean())
    return period if MIN_PERIOD <= period <= MAX_PERIOD else np.nan


def frequency_map(img: np.ndarray, theta: np.ndarray, mask: np.ndarray | None = None,
                  block: int = 16, length: int = 48, width: int = 16) -> np.ndarray:
    """Per-block ridge frequency (cycles/pixel), expanded to image size.

    Invalid blocks are filled with the median of valid ones. Crude on purpose;
    smoothed interpolation is better (exercise).
    """
    h, w = img.shape
    hb, wb = h // block, w // block
    freq = np.full((hb, wb), np.nan, dtype=np.float32)
    for by in range(hb):
        for bx in range(wb):
            cy, cx = by * block + block // 2, bx * block + block // 2
            if mask is not None and not mask[cy, cx]:
                continue
            p = _block_period(img, cx, cy, float(theta[cy, cx]), length, width)
            if not np.isnan(p):
                freq[by, bx] = 1.0 / p

    valid = ~np.isnan(freq)
    fill = float(np.median(freq[valid])) if valid.any() else 0.1
    freq[~valid] = fill
    full = cv2.resize(freq, (wb * block, hb * block), interpolation=cv2.INTER_NEAREST)
    out = np.full((h, w), fill, dtype=np.float32)
    out[: hb * block, : wb * block] = full
    return out
