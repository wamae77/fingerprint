# fpkit — M1: classical fingerprint pipeline

Goal of M1: build segmentation → orientation → frequency → Gabor enhancement →
binarization → thinning → minutiae → matching by hand, and measure EER on FVC2002
with the official protocol. No deep learning.

## Setup
    python -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"
    pytest -q
    python scripts/enhance_demo.py                 # synthetic sanity check
    python scripts/enhance_demo.py path/to/1_1.tif # real image

## Status
| Stage | Module | Verified on |
|---|---|---|
| Segmentation | `segment.py` | visual only (exercise 3) |
| Orientation field + coherence | `orientation.py` | synthetic ground truth |
| Ridge frequency | `frequency.py` | synthetic ground truth |
| Gabor enhancement | `gabor.py` | synthetic ground truth |
| FVC protocol + EER/FMR100/FMR1000/ZeroFMR | `eval/` | known distributions |
| Binarization, thinning | — | **you, next** |
| Minutiae (crossing number) + cleanup | — | **you** |
| Matcher | — | **you** |

Measured on concentric rings, 9px period (median orientation error / estimated period /
correlation with clean image before → after enhancement):

| noise σ | orient. err | period | corr before | corr after |
|---|---|---|---|---|
| 0.5 | 0.7° | 9.0 | 0.58 | 0.99 |
| 1.0 | 2.3° | 9.0 | 0.34 | 0.95 |
| 1.5 | 5.1° | 9.0 | 0.23 | 0.87 |

## Data
FVC2002 has four DBs. The small "B" sets (10 fingers × 8) are the easy starting
point; put them under `data/FVC2002/DB1_B/`. The full "A" sets (100 × 8) are what
published EERs use. Research use only.

## Exercises (do these before writing new stages)
1. **Break the orientation estimator on purpose.** Sweep `block_sigma` 2→15 on a real
   print. Watch cores/deltas blur at high values and noise dominate at low ones.
   Pick a value and write down why.
2. **Frequency fill.** Replace the median fill for invalid blocks with a Gaussian-
   weighted interpolation of valid neighbours. Does enhancement near the finger
   edge improve?
3. **Segmentation ground truth.** Hand-mask 10 FVC images (any tool), compute IoU of
   `segment()`. Find the image where it fails worst and explain the failure.
4. **Gabor σ.** `sigma_x` controls ridge-break healing vs creation of fake ridges.
   Try 2, 4, 6 on a scarred print. You'll see enhancement *invent* structure — that
   invented structure becomes false minutiae later.
5. **Speed.** The demo takes ~1s per image, mostly Python loops in `frequency_map`
   and repeated full-image filtering. Profile, then vectorize or crop filtering to
   mask bounding boxes. Target <150ms at 388×374.

## Next stages (spec)
- `binarize.py`: threshold enhanced image at 0 inside mask; smooth with a small
  median filter.
- `thin.py`: `skimage.morphology.skeletonize`, then write Zhang-Suen yourself once
  and compare.
- `minutiae.py`: crossing number on the skeleton (CN=1 ending, CN=3 bifurcation);
  angle from ridge tracing ~10px; remove minutiae near mask border, short spurs,
  bridges, and pairs closer than ~half a ridge period with opposing directions.
- `match.py`: start with Hough-based alignment (vote over dx, dy, dθ), count paired
  minutiae within distance/angle tolerance, normalize. Then implement MCC.
- `scripts/eval_fvc.py`: run protocol, print `Report`, save DET plot.
  Reference: SourceAFIS (Java) on the same pairs.
