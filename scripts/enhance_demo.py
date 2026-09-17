"""Run the enhancement pipeline on one image and save a debug panel.

    python scripts/enhance_demo.py                      # synthetic noisy rings
    python scripts/enhance_demo.py data/FVC2002/DB1_B/101_1.tif
"""
import sys
import time
import cv2
import numpy as np
from fpkit.io import load_gray
from fpkit.segment import segment
from fpkit.orientation import orientation_field
from fpkit.frequency import frequency_map
from fpkit.gabor import enhance
from fpkit.synthetic import concentric_rings
from fpkit.viz import to_u8, draw_orientation


def main():
    if len(sys.argv) > 1:
        img = load_gray(sys.argv[1])
        mask = segment(img)
        out_path = "debug_" + sys.argv[1].replace("/", "_").rsplit(".", 1)[0] + ".png"
    else:
        img, _, _ = concentric_rings(noise=1.0)
        img = (img - img.min()) / (img.max() - img.min())
        mask = np.ones_like(img, dtype=bool)
        out_path = "debug_synthetic.png"

    t0 = time.perf_counter()
    theta, coh = orientation_field(img)
    freq = frequency_map(img, theta, mask)
    enh = enhance(img, theta, freq, mask)
    ms = (time.perf_counter() - t0) * 1000

    period = 1.0 / np.median(freq[mask])
    print(f"{img.shape} | median ridge period {period:.1f}px | "
          f"mean coherence {coh[mask].mean():.2f} | {ms:.0f} ms")

    panels = [
        cv2.cvtColor(to_u8(img), cv2.COLOR_GRAY2BGR),
        draw_orientation(img, theta, mask),
        cv2.applyColorMap(to_u8(coh * mask), cv2.COLORMAP_VIRIDIS),
        cv2.cvtColor(to_u8(enh), cv2.COLOR_GRAY2BGR),
    ]
    labels = ["input", "orientation", "coherence", "gabor enhanced"]
    for p, label in zip(panels, labels):
        cv2.putText(p, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imwrite(out_path, np.hstack(panels))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
