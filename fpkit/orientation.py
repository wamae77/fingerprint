"""Ridge orientation field via least-squares gradient averaging.

Why not just average gradient angles? Orientation is axial: 1 deg and 181 deg
are the same ridge direction, and naive averaging cancels them out. Doubling
the angle (working with Gxx, Gyy, Gxy) maps theta and theta+pi to the same
point, so averaging becomes well-defined.

Convention used everywhere in fpkit:
    image coords: x -> right, y -> down
    theta = RIDGE direction, in [0, pi)
"""
import cv2
import numpy as np


def orientation_field(img: np.ndarray, grad_sigma: float = 1.0,
                      block_sigma: float = 7.0) -> tuple[np.ndarray, np.ndarray]:
    """Dense (per-pixel) ridge orientation and coherence.

    grad_sigma:  pre-smoothing before derivatives (noise suppression)
    block_sigma: averaging window; larger = smoother field but blurs
                 singular points (cores/deltas). Experiment with it.

    Returns (theta, coherence). Coherence in [0, 1]: 1 = perfectly parallel
    ridges, ~0 = no dominant direction (noise, scars, singular points).
    """
    smoothed = cv2.GaussianBlur(img, (0, 0), grad_sigma)
    gx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)

    gxx = cv2.GaussianBlur(gx * gx, (0, 0), block_sigma)
    gyy = cv2.GaussianBlur(gy * gy, (0, 0), block_sigma)
    gxy = cv2.GaussianBlur(gx * gy, (0, 0), block_sigma)

    # dominant GRADIENT direction; ridges run perpendicular to it
    grad_dir = 0.5 * np.arctan2(2 * gxy, gxx - gyy)
    theta = np.mod(grad_dir + np.pi / 2, np.pi)

    coherence = np.sqrt((gxx - gyy) ** 2 + 4 * gxy ** 2) / (gxx + gyy + 1e-8)
    return theta.astype(np.float32), coherence.astype(np.float32)


def angular_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest difference between two axial angles, in [0, pi/2]."""
    d = np.mod(a - b, np.pi)
    return np.minimum(d, np.pi - d)
