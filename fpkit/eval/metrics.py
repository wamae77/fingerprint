"""Verification metrics. Convention: higher score = more similar.

FMR  (false match rate):     impostor pairs scoring >= threshold
FNMR (false non-match rate): genuine pairs scoring  <  threshold
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Report:
    eer: float
    fmr100: float     # lowest FNMR with FMR <= 1%
    fmr1000: float    # lowest FNMR with FMR <= 0.1%
    zero_fmr: float   # lowest FNMR with FMR == 0
    n_genuine: int
    n_impostor: int

    def __str__(self):
        return (f"EER {self.eer:.2%} | FMR100 {self.fmr100:.2%} | "
                f"FMR1000 {self.fmr1000:.2%} | ZeroFMR {self.zero_fmr:.2%} | "
                f"G={self.n_genuine} I={self.n_impostor}")


def det_curve(genuine, impostor):
    g, i = np.asarray(genuine, float), np.asarray(impostor, float)
    thresholds = np.unique(np.concatenate([g, i, [np.inf]]))
    g_sorted, i_sorted = np.sort(g), np.sort(i)
    fnmr = np.searchsorted(g_sorted, thresholds, side="left") / len(g)
    fmr = 1.0 - np.searchsorted(i_sorted, thresholds, side="left") / len(i)
    return thresholds, fmr, fnmr


def evaluate(genuine, impostor) -> Report:
    _, fmr, fnmr = det_curve(genuine, impostor)
    k = int(np.argmin(np.abs(fmr - fnmr)))
    eer = float((fmr[k] + fnmr[k]) / 2)

    def fnmr_at(max_fmr):
        ok = fmr <= max_fmr
        return float(fnmr[ok].min()) if ok.any() else 1.0

    return Report(eer, fnmr_at(0.01), fnmr_at(0.001), fnmr_at(0.0),
                  len(genuine), len(impostor))
