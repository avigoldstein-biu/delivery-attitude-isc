"""
Shared functions for Study 2 (listener attitude x social context, one crossed cohort).
Two crossed manipulations were run in the same participants: ATTITUDE (Positive/Negative
description before an identical lecture) and CUE (audience silhouettes present/absent on a
second, different lecture). Imported by the numbered study2 scripts.

Statistical note: between-group ISC is tested at the SUBJECT level (each participant contributes
one leave-one-out ISC value per region/band), which avoids the non-independence of pairwise ISC.
"""
import os, glob, numpy as np, pandas as pd, scipy.io as sio
from scipy import stats

# ---------------------------------------------------------------- CONFIG (edit paths)
SRC   = r"G:/Barak1/iscex/source_iscex_deposit"         # per-subject ISCex_<n>.mat (fields S2_band, S3_band: 72 x T x 5)
ATLAS = r"G:/Barak1/iscex/ISCex_1/atlas_info.mat"       # AAL tissue map -> 72 region labels
META  = r"G:/Barak1/iscex/trial_metadata.csv"           # subj, include_S2/S3, attitude_grp, cue_grp
RNG   = np.random.RandomState(0)
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]

labels = [str(x).strip("'") for x in sio.loadmat(ATLAS, squeeze_me=True)["tissuelabel"]]
meta = pd.read_csv(META)
# right-perisylvian ROI (the Study-1 delivery core / Study-2 attitude network)
PERISYLVIAN = ["Rolandic_Oper_R", "Temporal_Sup_R", "Temporal_Mid_R", "Insula_R", "Frontal_Inf_Oper_R"]
peri_idx = [labels.index(r) for r in PERISYLVIAN]

def loo(A):
    """Leave-one-out ISC. A (N_subj, N_region, T) -> (N_subj, N_region) Fisher-z."""
    N = A.shape[0]; tot = A.sum(0); out = np.zeros((N, A.shape[1]))
    for i in range(N):
        o = (tot-A[i])/(N-1); a = A[i]-A[i].mean(-1, keepdims=True); oo = o-o.mean(-1, keepdims=True)
        r = (a*oo).sum(-1)/(np.sqrt((a**2).sum(-1)*(oo**2).sum(-1))+1e-12)
        out[i] = np.arctanh(np.clip(r, -.999, .999))
    return out

def fdr(p, q=0.05):
    p = np.asarray(p); o = np.argsort(p); n = len(p); thr = q*np.arange(1, n+1)/n
    below = p[o] <= thr; k = np.where(below)[0].max() if below.any() else -1
    sig = np.zeros(n, bool)
    if k >= 0: sig[o[:k+1]] = True
    return sig

def load(field, inc, gcol):
    """Load per-subject band data for an experiment. field in {S2_band, S3_band}.
    Returns X (N,72,T,5), grp (N,), subj_ids (N,). LOO is computed within group."""
    subs = sorted(glob.glob(os.path.join(SRC, "ISCex_*.mat")), key=lambda p: int(os.path.basename(p)[6:-4]))
    rows = [(int(os.path.basename(s)[6:-4]), s) for s in subs]
    rows = [(n, s, str(meta[meta.subj == n][gcol].iloc[0])) for n, s in rows
            if not meta[meta.subj == n].empty and int(meta[meta.subj == n][inc].iloc[0]) == 1]
    grp = np.array([g for _, _, g in rows]); ids = np.array([n for n, _, _ in rows])
    dat = [sio.loadmat(s)[field] for _, s, _ in rows]; T = min(d.shape[1] for d in dat)
    X = np.stack([d[:, :T, :].astype(np.float32) for d in dat])
    return X, grp, ids

NPERM = 10000                                   # permutations for the between-group test (Section 2.5)

def unit(x):                                    # (N,R,T) -> zero-mean, unit-norm over T
    x = x - x.mean(-1, keepdims=True)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)

def loo_from_G(G, idx):
    """LOO-ISC (Fisher-z) for the subjects in idx, from a single ISC Gram matrix G (N,N)."""
    n = len(idx); Gg = G[np.ix_(idx, idx)]; rs = Gg.sum(1)
    num = (rs - 1.0) / (n - 1); Sg = Gg.sum(); m2 = (Sg - 2*rs + 1.0) / (n - 1)**2
    r = num / np.sqrt(np.clip(m2, 1e-12, None))
    return np.arctanh(np.clip(r, -.999, .999))

def _loo_mean_batch(G3, idx):
    """Mean LOO-ISC over `idx`, vectorised across regions. G3 (72,N,N), idx (n,) -> (72,)."""
    n = len(idx); Gg = G3[:, idx][:, :, idx]                       # (72,n,n)
    rs = Gg.sum(2); num = (rs - 1.0) / (n - 1)                     # (72,n)
    Sg = Gg.sum((1, 2))[:, None]; m2 = (Sg - 2*rs + 1.0) / (n - 1)**2
    r = num / np.sqrt(np.clip(m2, 1e-12, None))
    return np.arctanh(np.clip(r, -.999, .999)).mean(1)            # (72,)

def perm_between(X_band, gA, gB, n_perm=NPERM, rng=None):
    """Valid between-group ISC test for one band. X_band (N,72,T) raw region signals; gA,gB are the
    index arrays of the two groups. For each permutation the group labels are shuffled and the
    leave-one-out ISC is RE-COMPUTED within the permuted groups (via the region ISC Gram matrix; the
    permutation is vectorised across the 72 regions).

    Returns obs (72,), the two-sided p (72,) and the one-sided p (72,) for A > B. Both p values are
    returned because the paper reports them separately: the regions surviving FDR without a
    directional assumption come from the two-sided p, and the further regions that survive only
    under the a priori direction from the one-sided p.

    `rng` controls the permutation stream; the module-level RNG is used if it is not given."""
    r = RNG if rng is None else rng
    Xb = unit(X_band); G3 = np.einsum("nrt,mrt->rnm", Xb, Xb)      # (72,N,N) per-region ISC matrix
    N = Xb.shape[0]; nA = len(gA)
    obs = _loo_mean_batch(G3, gA) - _loo_mean_batch(G3, gB)
    null = np.zeros((n_perm, 72))
    for k in range(n_perm):
        pm = r.permutation(N); null[k] = _loo_mean_batch(G3, pm[:nA]) - _loo_mean_batch(G3, pm[nA:])
    return obs, (np.abs(null) >= np.abs(obs)).mean(0), (null >= obs).mean(0)

def perm_set(X_band, gA, gB, idx, n_perm=NPERM, rng=None):
    """Section 2.6's second test: the group difference in the leave-one-out ISC averaged across a
    SET of parcels, against a null constructed over participants.

    Each participant contributes one value per band -- their leave-one-out correlation averaged
    across the parcels in `idx` -- and the group labels are permuted with the leave-one-out ISC
    re-computed within the permuted groups, as everywhere else. This asks whether the
    region-averaged difference departs from zero, which is a different question from the enrichment
    test of roi_stage_grid.py: that one asks whether an effect is more concentrated in this region
    than elsewhere in the brain, and does not test against zero at all.

    Only the parcels in `idx` enter the Gram matrix, so this is much cheaper than perm_between.

    Returns obs, two-sided p, one-sided p (A > B)."""
    r = RNG if rng is None else rng
    Xb = unit(X_band[:, idx])                                  # (N,k,T)
    G3 = np.einsum("nrt,mrt->rnm", Xb, Xb)                     # (k,N,N)
    N = Xb.shape[0]; nA = len(gA)
    obs = float((_loo_mean_batch(G3, gA) - _loo_mean_batch(G3, gB)).mean())
    null = np.zeros(n_perm)
    for k in range(n_perm):
        pm = r.permutation(N)
        null[k] = float((_loo_mean_batch(G3, pm[:nA]) - _loo_mean_batch(G3, pm[nA:])).mean())
    return obs, float((np.abs(null) >= abs(obs)).mean()), float((null >= obs).mean())

def perm_global(X_band, gA, gB, n_perm=NPERM, one_sided=True, rng=None):
    """Global test: the paper's 'global ISC' is the MEAN over the 72 regions of the per-region
    leave-one-out ISC. Same group-label permutation (LOO recomputed within permuted groups)."""
    r = RNG if rng is None else rng
    Xb = unit(X_band); G3 = np.einsum("nrt,mrt->rnm", Xb, Xb)
    N = Xb.shape[0]; nA = len(gA)
    obs = float((_loo_mean_batch(G3, gA) - _loo_mean_batch(G3, gB)).mean())
    null = np.zeros(n_perm)
    for k in range(n_perm):
        pm = r.permutation(N)
        null[k] = (_loo_mean_batch(G3, pm[:nA]) - _loo_mean_batch(G3, pm[nA:])).mean()
    p = float((null >= obs).mean()) if one_sided else float((np.abs(null) >= np.abs(obs)).mean())
    base = float(_loo_mean_batch(G3, np.arange(N)).mean())
    return obs, p, base
