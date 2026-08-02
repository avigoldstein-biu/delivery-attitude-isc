"""
The second of the two region-of-interest tests described in Section 2.6.

roi_stage_grid.py runs the first: the ENRICHMENT test, which asks whether an effect is more
concentrated in the four-parcel right auditory-perisylvian region than in same-size sets drawn from
elsewhere in the atlas. That test is constructed over regions, is conditional on the observed map,
and does not test the region-averaged effect against zero.

This script runs the second: the AGAINST-ZERO test, constructed over participants. Each participant
contributes one value per band -- their leave-one-out correlation averaged across the four parcels
-- and the group difference is tested with the group-label permutation of Section 2.5, with
leave-one-out ISC re-computed within the permuted groups. It is what Section 4.3 means by "did not
differ from zero", and it is the source of the beta value reported there (+0.031, p = .070 after
feature regression).

The two answer different questions and the paper reports them separately throughout. A contrast can
be concentrated in this cortex without the region-averaged difference departing from zero, and the
Study 2 beta effect is exactly that case.

Every contrast is run at all three control stages, so the grid matches Table S7.

CONSISTENCY CHECK. The staging here -- feature regression, and the aperiodic removal applied before
it -- repeats what roi_stage_grid.py does when it builds the effect maps. To make sure the two
cannot drift apart, the region-averaged effects computed here are compared against
<DERIV>/s23_stage_maps.npz and a warning is printed if they disagree. Run roi_stage_grid.py first.

Requires: source_iscex/, the Study 2 feature batteries from study2_build_features.py, and
<DERIV>/s23_stage_maps.npz from roi_stage_grid.py.

Usage: python roi_against_zero.py
"""
import glob, os, sys
import numpy as np
import scipy.io as sio

from lib import periodic                                   # same aperiodic removal Study 1 uses
from lib2 import SRC, meta, labels, perm_set, NPERM

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"G:/Barak1"
DERIV = os.path.join(ROOT, "reanalysis")   # reconstruction outputs and intermediate results; point anywhere

FEAT_DIR = os.path.join(ROOT, "iscex", "features")
CACHE = os.path.join(DERIV, "s23_stage_maps.npz")
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
ROI = ["Rolandic_Oper_R", "Temporal_Sup_R", "Temporal_Mid_R", "Insula_R"]
roi = [labels.index(r) for r in ROI]
TOL = 5e-4                                                 # agreement required with the cached maps

RNG = np.random.RandomState(7)


def rows_for(inc, gcol):
    subs = sorted(glob.glob(os.path.join(SRC, "ISCex_*.mat")),
                  key=lambda p: int(os.path.basename(p)[6:-4]))
    r = [(int(os.path.basename(s)[6:-4]), s) for s in subs]
    return [(n, s, str(meta[meta.subj == n][gcol].iloc[0])) for n, s in r
            if not meta[meta.subj == n].empty and int(meta[meta.subj == n][inc].iloc[0]) == 1]


def run(field, inc, gcol, A, B, bands, featmap, tag, title, cached):
    rows = rows_for(inc, gcol); bi = [BANDS.index(b) for b in bands]
    raw, per = [], []
    for k, (n, s, g) in enumerate(rows):
        d = sio.loadmat(s)[field].astype(np.float32)       # (72,T,5): all bands needed for the 1/f fit
        raw.append(d[:, :, bi])
        per.append(periodic(d[None]).astype(np.float32)[0][:, :, bi])
        if (k + 1) % 20 == 0: print(f"  {tag} loaded {k+1}/{len(rows)}", flush=True)
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

    print(f"\n===== {title} =====  n={len(rows)} ({A}={len(gA)}, {B}={len(gB)}), "
          f"{NPERM:,} permutations")
    print(f"  {'band':6s} {'stage':6s} {'ROI mean d':>11s} {'p one-sided':>12s} {'p two-sided':>12s} "
          f"{'vs cached map':>14s}")
    for j, b in enumerate(bands):
        for stage, X in (("raw", R[:, :, :, j]),
                         ("feat", regress(R[:, :, :, j])),
                         ("both", regress(P[:, :, :, j]))):
            obs, p2, p1 = perm_set(X, gA, gB, roi, rng=RNG)
            key = f"{tag}_{b}_{stage}"
            chk = "  (not cached)"
            if key in cached:
                ref = float(np.asarray(cached[key])[roi].mean())
                chk = f"{ref:+14.4f}" if abs(ref - obs) <= TOL else f"  MISMATCH {ref:+.4f}"
            print(f"  {b:6s} {stage:6s} {obs:+11.4f} {p1:12.4f} {p2:12.4f} {chk}", flush=True)


cached = dict(np.load(CACHE, allow_pickle=True)) if os.path.exists(CACHE) else {}
if not cached:
    print(f"note: {CACHE} not found, so the consistency check is skipped; run roi_stage_grid.py first\n")

run("S2_band", "include_S2", "attitude_grp", "Positive", "Negative",
    ["theta", "alpha", "beta"], {"*": "S2_feat100.npy"}, "S2",
    "Study 2: listener attitude (positive - negative)", cached)
run("S3_band", "include_S3", "cue_grp", "With", "Without",
    ["alpha", "beta"], {"With": "S3_cue_feat100.npy", "Without": "S3_nocue_feat100.npy"}, "S3",
    "Study 2: audience cue (cue - no cue)", cached)

print("\nThis is the test against zero. Concentration within the region is a separate question and\n"
      "is answered by roi_stage_grid.py; the two are reported separately (Section 2.6).")
