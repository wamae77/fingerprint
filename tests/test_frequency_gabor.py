import numpy as np
from fpkit.frequency import frequency_map
from fpkit.gabor import enhance
from fpkit.orientation import orientation_field
from fpkit.synthetic import concentric_rings


import pytest


@pytest.mark.parametrize("noise", [0.3, 1.0, 1.5])
def test_frequency_recovers_period(noise):
    img, _, _ = concentric_rings(period=9.0, noise=noise)
    theta, _ = orientation_field(img)
    f = frequency_map(img, theta)
    period = 1.0 / np.median(f[40:-40, 40:-40])
    assert abs(period - 9.0) < 0.5, period


def test_gabor_improves_noisy_image():
    img, clean, _ = concentric_rings(noise=1.0)
    theta, _ = orientation_field(img)
    f = frequency_map(img, theta)
    out = enhance(img, theta, f)
    s = slice(40, -40)

    def corr(a, b):
        return np.corrcoef(a[s, s].ravel(), b[s, s].ravel())[0, 1]

    before, after = corr(img, clean), corr(out, clean)
    assert after > before + 0.4, (before, after)
