"""
03 - Aperiodic control for the attitude ISC effect. Remove the aperiodic (1/f) component from the
5-band amplitude spectrum at each timepoint (reusing the Study-1 lib.periodic), then recompute the
Positive vs Negative ISC contrast on the periodic (oscillatory) component.


"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib import periodic                                   # reuse the validated aperiodic removal
from lib2 import load, perm_between, fdr, BANDS

X, grp, _ = load("S2_band", "include_S2", "attitude_grp")
per = periodic(X).astype(np.float32)                        # (N,72,T,5) periodic component
gA = np.where(grp == "Positive")[0]; gB = np.where(grp == "Negative")[0]

def contrast(A):                                           # valid Positive>Negative test, per region
    obs, _, p1 = perm_between(A, gA, gB); return obs, fdr(p1)   # directional: the effect is predicted

print(f"S2 attitude on the PERIODIC (1/f-removed) component (n={len(grp)}):")
print(f"  {'band':6s} {'raw global d':>12s} {'raw FDR':>8s} | {'periodic d':>11s} {'periodic FDR':>13s}")
for bi, b in enumerate(BANDS):
    dr, sr = contrast(X[:, :, :, bi]); dp, sp = contrast(per[:, :, :, bi])
    print(f"  {b:6s} {dr.mean():+12.4f} {sr.sum():8d} | {dp.mean():+11.4f} {sp.sum():13d}")
