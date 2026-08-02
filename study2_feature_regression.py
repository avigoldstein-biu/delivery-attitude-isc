"""
02 - Is the attitude ISC effect stimulus-driven? Regress a low-level acoustic/visual feature
battery out of each region's time course before recomputing the Positive vs Negative ISC contrast,
with a phase-scrambled-feature null (matched spectrum, no true stimulus locking) to control for the
degrees of freedom removed.

Features are the MEG-aligned 10-feature battery from study2_build_features.py (features/<tag>_feat100.npy).

Reports at both stages (raw, and after the feature regression):
  - the global (72-region mean) group difference and its permutation p
  - how much of the raw global difference the regression leaves, against the phase-scrambled null
  - the number of regions surviving FDR, two-sided and one-sided
  - the mean effect over the four-parcel region of interest of Section 2.6, and the effect and
    uncorrected p at the right rolandic operculum, which carries the focal beta difference

The group test is the one study2_isc_stats.py uses: group labels permuted with leave-one-out ISC
re-computed within the permuted groups. An independent-samples t-test on the leave-one-out values
held fixed is NOT equivalent, because those values are computed against a group mean that each
participant contributes to and so are not independent observations. The two agree closely on a
single region's uncorrected p but diverge badly on the 72-region mean, where the dependence
compounds, and on any FDR count.

Note that the region of interest here is the four parcels of Section 2.6. lib2.PERISYLVIAN is a
five-parcel set that additionally includes the right inferior frontal operculum, which Section 4.3
reports in the whole-brain analysis but excludes from the region-level tests.

Runtime is dominated by loading the region time courses and by the permutations; expect tens of
minutes.

Usage: python study2_feature_regression.py
"""
import os, sys
import numpy as np
from lib2 import load, loo, fdr, perm_between, perm_global, labels, BANDS, NPERM

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
FEAT_DIR = r"G:/Barak1/iscex/features"          # from study2_build_features.py
RNG = np.random.RandomState(0)                  # phase-scramble draw
RNG_TEST = np.random.RandomState(31)            # permutation stream for the group tests

ROI = ["Rolandic_Oper_R", "Temporal_Sup_R", "Temporal_Mid_R", "Insula_R"]   # Section 2.6
roi = [labels.index(r) for r in ROI]
ROP = labels.index("Rolandic_Oper_R")


def resid(X, M):                                # regress feature matrix M (T,k) out of X (N,T)
    Mp = np.linalg.pinv(M); return X - (X @ Mp.T) @ M.T


def scramble(M):                                # phase-scramble each feature (matched spectrum)
    out = np.zeros_like(M)
    for k in range(M.shape[1]):
        F = np.fft.rfft(M[:, k]); ph = RNG.uniform(0, 2*np.pi, len(F)); ph[0] = 0
        out[:, k] = np.fft.irfft(np.abs(F)*np.exp(1j*ph), n=M.shape[0])
    return (out-out.mean(0))/(out.std(0)+1e-9)


def regress_all(Xb, M):                         # (N,72,T) -> feature residual, region by region
    return np.stack([resid(Xb[:, r], M) for r in range(72)], 1)


X, grp, _ = load("S2_band", "include_S2", "attitude_grp")
F = np.load(os.path.join(FEAT_DIR, "S2_feat100.npy"))          # (T,10) aligned, 0.8 Hz LP, z
T = min(X.shape[2], len(F)); X = X[:, :, :T]; F = F[:T]; Fn = scramble(F)
pos = grp == "Positive"; neg = grp == "Negative"
gA = np.where(pos)[0]; gB = np.where(neg)[0]


def contrast(A):                                              # global Positive-Negative, per band
    iz = loo(A)
    return iz[pos].mean(0) - iz[neg].mean(0)


print(f"S2 attitude, feature regression (n={len(grp)}, T={T}, {NPERM:,} permutations)")
print("\nglobal d ISC (Positive - Negative), and what the regression leaves:")
print(f"  {'band':6s} {'raw':>8s} {'+feature':>9s} {'+scrambled':>11s}  (retained%)")
resid_cache = {}
for bi, b in enumerate(BANDS):
    raw = contrast(X[:, :, :, bi]).mean()
    Xr = regress_all(X[:, :, :, bi], F); resid_cache[b] = Xr
    res = contrast(Xr).mean()
    nul = contrast(regress_all(X[:, :, :, bi], Fn)).mean()
    print(f"  {b:6s} {raw:+8.4f} {res:+9.4f} {nul:+11.4f}   ({100*res/raw if raw else 0:3.0f}%)",
          flush=True)

print("\nby stage, tested with the group-label permutation of Section 2.5:")
print(f"  {'band':6s} {'stage':6s} {'global d':>9s} {'global p':>9s} {'FDR 2s':>7s} {'FDR 1s':>7s} "
      f"{'ROI mean':>9s} {'RolOperR':>9s} {'RolOper p1':>11s}")
for bi, b in enumerate(BANDS):
    for stage, A in (("raw", X[:, :, :, bi]), ("feat", resid_cache[b])):
        d, p, _ = perm_global(A, gA, gB, rng=RNG_TEST)
        obs, p2, p1 = perm_between(A, gA, gB, rng=RNG_TEST)
        print(f"  {b:6s} {stage:6s} {d:+9.4f} {p:9.4f} {int(fdr(p2).sum()):7d} {int(fdr(p1).sum()):7d} "
              f"{obs[roi].mean():+9.4f} {obs[ROP]:+9.4f} {p1[ROP]:11.4f}", flush=True)

print("\nAn FDR count near a threshold is seed-sensitive: a region whose p falls within Monte Carlo\n"
      "error of its Benjamini-Hochberg threshold enters or leaves the set from run to run. Exact\n"
      "counts should be read with that in mind (Supplementary S6).")
