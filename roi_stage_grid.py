"""
ROI enrichment by control stage -- Supplementary Table S7.

The convergence test compares the mean effect in the four-parcel right auditory-perisylvian ROI
against a null of same-size region sets. Study 1's surviving core is defined after both controls
(feature regression applied to the aperiodic-removed component), whereas the Study 2 and 3
contrasts are reported after feature regression alone. This script produces every cell of that
grid so the choice of stage is visible rather than implicit.

WHY THE STAGES DIFFER. The aperiodic step retains only 13-51% of baseline inter-subject
correlation in delta through beta and inflates gamma, the top edge of the fit, by about 30%
(see reanalysis/aperiodic_diagnostics/). A within-subject paired contrast absorbs that loss --
Study 1's beta effect is large enough to survive it with magnitude and localisation intact -- but
a between-group contrast at a quarter of the effect size does not: no region survives correction
in any band in Study 2 or 3 at that stage, and the audience-cue control, null beforehand, reaches
nominal significance afterwards. The aperiodic stage is therefore reported for Study 1 and the
feature stage for Studies 2 and 3.

The enrichment null enumerates all 1,028,790 four-parcel sets, so each p is exact.

Study 2 and 3 effect maps are computed once and cached to <DERIV>/s23_stage_maps.npz; delete that
file to recompute. Study 1's three maps are read from the Study 1 result file directly.

Requires: source_iscex/ (Study 2 and 3 region time courses), the Study 2/3 feature batteries from
study2_build_features.py, and study1_source_myica3_greater_pc0.npz from study1_isc.py.

Usage: python roi_stage_grid.py
"""
import glob, os, sys
from itertools import combinations, islice
import numpy as np
import scipy.io as sio

from lib import periodic                                   # same aperiodic removal Study 1 uses
from lib2 import SRC, meta, labels, unit, _loo_mean_batch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"G:/Barak1"
DERIV = os.path.join(ROOT, "reanalysis")   # reconstruction outputs and intermediate results; point anywhere

FEAT_DIR = os.path.join(ROOT, "iscex", "features")
CACHE = os.path.join(DERIV, "s23_stage_maps.npz")
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
ROI = ["Rolandic_Oper_R", "Temporal_Sup_R", "Temporal_Mid_R", "Insula_R"]
roi = [labels.index(r) for r in ROI]


def rows_for(inc, gcol):
    subs = sorted(glob.glob(os.path.join(SRC, "ISCex_*.mat")),
                  key=lambda p: int(os.path.basename(p)[6:-4]))
    r = [(int(os.path.basename(s)[6:-4]), s) for s in subs]
    return [(n, s, str(meta[meta.subj == n][gcol].iloc[0])) for n, s in r
            if not meta[meta.subj == n].empty and int(meta[meta.subj == n][inc].iloc[0]) == 1]


def dmap(X, gA, gB):
    """72-region effect map, leave-one-out recomputed within each group (as in study2_isc_stats)."""
    Xu = unit(X); G3 = np.einsum("nrt,mrt->rnm", Xu, Xu)
    return _loo_mean_batch(G3, gA) - _loo_mean_batch(G3, gB)


def build_maps(field, inc, gcol, A, B, bands, featmap, tag, out):
    rows = rows_for(inc, gcol); bi = [BANDS.index(b) for b in bands]
    raw, per = [], []
    for k, (n, s, g) in enumerate(rows):
        d = sio.loadmat(s)[field].astype(np.float32)       # (72,T,5): all bands needed for the 1/f fit
        raw.append(d[:, :, bi])
        per.append(periodic(d[None]).astype(np.float32)[0][:, :, bi])
        if (k + 1) % 20 == 0: print(f"  {tag} {k+1}/{len(rows)}", flush=True)
    T = min(x.shape[1] for x in raw)
    F = {k: np.load(os.path.join(FEAT_DIR, v)) for k, v in featmap.items()}
    T = min([T] + [len(f) for f in F.values()])
    R = np.stack([x[:, :T, :] for x in raw]); P = np.stack([x[:, :T, :] for x in per])
    del raw, per
    grp = np.array([g for _, _, g in rows])
    gA = np.where(grp == A)[0]; gB = np.where(grp == B)[0]
    Fi = {k: (np.linalg.pinv(v[:T]), v[:T]) for k, v in F.items()}

    def regress(X):                                        # each group's own stimulus features
        Y = np.empty_like(X)
        for i, g in enumerate(grp):
            Mp, M = Fi[g if g in Fi else "*"]
            Y[i] = X[i] - (X[i] @ Mp.T) @ M.T
        return Y

    for j, b in enumerate(bands):
        out[f"{tag}_{b}_raw"] = dmap(R[:, :, :, j], gA, gB)
        out[f"{tag}_{b}_feat"] = dmap(regress(R[:, :, :, j]), gA, gB)
        out[f"{tag}_{b}_both"] = dmap(regress(P[:, :, :, j]), gA, gB)
        print(f"  {tag} {b}: ROI mean raw {out[f'{tag}_{b}_raw'][roi].mean():+.4f} -> "
              f"feat {out[f'{tag}_{b}_feat'][roi].mean():+.4f} -> "
              f"both {out[f'{tag}_{b}_both'][roi].mean():+.4f}", flush=True)


if os.path.exists(CACHE):
    m = dict(np.load(CACHE, allow_pickle=True)); print(f"Study 2/3 maps loaded from {CACHE}")
else:
    m = {}
    build_maps("S2_band", "include_S2", "attitude_grp", "Positive", "Negative",
               ["theta", "alpha", "beta"], {"*": "S2_feat100.npy"}, "S2", m)
    build_maps("S3_band", "include_S3", "cue_grp", "With", "Without",
               ["alpha", "beta"],
               {"With": "S3_cue_feat100.npy", "Without": "S3_nocue_feat100.npy"}, "S3", m)
    np.savez_compressed(CACHE, **m); print(f"Study 2/3 maps cached to {CACHE}")

s1 = np.load(os.path.join(ROOT, "paper_code", "study1_source_myica3_greater_pc0.npz"))
ALL = np.array(list(combinations(range(72), 4)))


def enrich(v, chunk=2_000_000):
    """Exact enrichment: mean effect in the ROI against every four-parcel set."""
    obs = v[roi].mean(); it = combinations(range(72), 4); k = n = 0
    while True:
        buf = list(islice(it, chunk))
        if not buf: break
        mm = v[np.array(buf, np.int8)].mean(1); k += int((mm >= obs).sum()); n += len(mm)
    return obs, k, (1 + k) / (n + 1)


GRID = [("Study 1, delivery", "beta", [np.asarray(s1["beta_raw_d"]), np.asarray(s1["beta_feat_d"]),
                                       np.asarray(s1["beta_both_d"])], 2),
        ("Study 2, attitude", "alpha", None, 1),
        ("Study 2, attitude", "beta",  None, 1),
        ("Study 2, attitude", "theta", None, 1),
        ("Study 3, cue",      "alpha", None, 1),
        ("Study 3, cue",      "beta",  None, 1)]

print(f"\nTABLE S7 -- ROI enrichment by control stage (exact over {len(ALL):,} four-parcel sets)")
print(f"{'contrast':20s} {'band':6s} {'raw':>22s} {'+ features':>22s} {'+ feat + aperiodic':>22s}")
for name, band, maps, primary in GRID:
    if maps is None:
        tag = "S2" if name.startswith("Study 2") else "S3"
        maps = [np.asarray(m[f"{tag}_{band}_{st}"]) for st in ("raw", "feat", "both")]
    cells = []
    for i, v in enumerate(maps):
        o, k, p = enrich(v)
        star = "*" if i == primary else " "
        cells.append(f"{o:+.4f} ({p:.5f}){star}")
    print(f"{name:20s} {band:6s} " + " ".join(f"{c:>22s}" for c in cells))
print("\n* = the stage reported in the main text. Study 1 is reported after both controls; the")
print("  Study 2 and 3 contrasts after feature regression -- see the module docstring for why.")
