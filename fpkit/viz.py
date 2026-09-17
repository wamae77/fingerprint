"""Debug visualizations. Look at these constantly: numbers lie, overlays don't."""
import cv2
import numpy as np


def to_u8(img: np.ndarray) -> np.ndarray:
    lo, hi = float(img.min()), float(img.max())
    return ((img - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)


def draw_orientation(img, theta, mask=None, step=12, length=10):
    vis = cv2.cvtColor(to_u8(img), cv2.COLOR_GRAY2BGR)
    h, w = theta.shape
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            if mask is not None and not mask[y, x]:
                continue
            dx = np.cos(theta[y, x]) * length / 2
            dy = np.sin(theta[y, x]) * length / 2
            cv2.line(vis, (int(x - dx), int(y - dy)), (int(x + dx), int(y + dy)),
                     (0, 0, 255), 1, cv2.LINE_AA)
    return vis
