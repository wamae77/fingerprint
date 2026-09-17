import numpy as np
from fpkit.orientation import orientation_field, angular_diff
from fpkit.synthetic import concentric_rings


def _interior(size, margin=40, center_radius=30):
    y, x = np.mgrid[0:size, 0:size]
    r = np.hypot(x - size / 2, y - size / 2)
    m = np.zeros((size, size), bool)
    m[margin:-margin, margin:-margin] = True
    return m & (r > center_radius)   # rings' center is a singular point


def test_orientation_clean():
    img, _, truth = concentric_rings(noise=0.0)
    theta, coh = orientation_field(img)
    m = _interior(img.shape[0])
    err = np.degrees(angular_diff(theta, truth))[m]
    assert np.median(err) < 2.0, np.median(err)
    assert coh[m].mean() > 0.8


def test_orientation_noisy():
    img, _, truth = concentric_rings(noise=0.8)
    theta, _ = orientation_field(img)
    m = _interior(img.shape[0])
    err = np.degrees(angular_diff(theta, truth))[m]
    assert np.median(err) < 5.0, np.median(err)
