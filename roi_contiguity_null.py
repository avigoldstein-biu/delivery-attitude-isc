"""
Contiguity-controlled null for the ROI enrichment test of cross_study_convergence.py.

That test compares the mean effect in the four-parcel right auditory-perisylvian ROI against a
null of random four-parcel sets. Sets drawn at random from the atlas are almost always scattered,
whereas the ROI is spatially contiguous. Adjacent parcels share signal through source leakage, so
their values are correlated and a contiguous set carries fewer effective independent values; the
mean over it has a wider sampling distribution than the mean over a scattered set. The
unrestricted null is therefore too narrow for a contiguous ROI, and part of the enrichment could
reward contiguity rather than location.

This script repeats the test against a null restricted to CONNECTED four-parcel sets. There are
few enough to enumerate completely, so each p is an exact proportion over the whole reference set
rather than an estimate from random draws; where no set in the reference exceeds the observed
value, the exceedance count is reported directly rather than a p of zero.

A second restriction additionally confines the null to the right hemisphere. This is a harder test
in a specific way: when an effect is right-lateralised, right-hemisphere comparison sets carry more
of it, the null shifts upward, and the ROI is less exceptional against it. The MORE lateralised an
effect is, the more it loses under this restriction -- so a near-symmetric effect is barely
affected while a strongly lateralised one weakens.

ADJACENCY. Two parcels adjoin if any pair of their grid points lies within a threshold distance.
It is computed on the TEMPLATE grid, not on any participant's anatomy: the source grid is a 1 cm
MNI lattice of shape DIM, and each participant's sourcemodel holds that same lattice warped
affinely into their own head, in the same point order. Working in lattice index units therefore
gives template geometry exactly, with 1 unit = 1 cm, and needs no sourcemodel at all. (Using one
participant's warped coordinates instead would impose that participant's affine scaling -- for the
first participant the 1 cm template step becomes 0.87, 0.99 and 0.78 cm along the three axes -- and
changes 4 of the 2556 parcel pairs at the 1.5 cm threshold.)

Two thresholds are reported: 1.0 cm admits face neighbours only, 1.5 cm also admits edge diagonals
(sqrt 2 = 1.41). The conclusion should not depend on which is used.

Each contrast is tested at its reported control stage: Study 1 after both controls, Studies 2 and
3 after feature regression. See roi_stage_grid.py for why the stages differ.

Requires: study1_source_myica3_greater_pc0.npz, <DERIV>/s23_stage_maps.npz (written by
roi_stage_grid.py) and atlas_info.mat. No individual anatomy.

Usage: python roi_contiguity_null.py
"""
import os, sys
from itertools import combinations
import numpy as np
import scipy.io as sio
from scipy.spatial.distance import cdist

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"G:/Barak1"
DERIV = os.path.join(ROOT, "reanalysis")   # reconstruction outputs and intermediate results; point anywhere

DIM = (17, 21, 18)              # template source-grid box, 1 cm isotropic; 17*21*18 = 6426 points
THRESHOLDS = [1.0, 1.5]         # cm: face neighbours only, and face + edge diagonals
ROI = ["Rolandic_Oper_R", "Temporal_Sup_R", "Temporal_Mid_R", "Insula_R"]

z = np.load(os.path.join(DERIV, "study2_isc_results.npz"), allow_pickle=True)   # study2_isc_stats.py
labels = [str(x).strip() for x in z["labels"]]
s1 = np.load(os.path.join(ROOT, "paper_code", "study1_source_myica3_greater_pc0.npz"))
# Each contrast at its REPORTED control stage (see roi_stage_grid.py for why the stages differ):
# Study 1 after both controls, Studies 2 and 3 after feature regression.
m23 = np.load(os.path.join(DERIV, "s23_stage_maps.npz"), allow_pickle=True)
tissue = np.asarray(sio.loadmat(os.path.join(ROOT, "Charisma", "atlas_info.mat"))["tissue"]).ravel()
lattice = np.array(np.unravel_index(np.arange(np.prod(DIM)), DIM, order="F")).T.astype(float)
parcel_pts = [lattice[tissue == r + 1] for r in range(72)]     # points in no parcel are NaN in `tissue`

roi = [labels.index(r) for r in ROI]
right = np.array([l.endswith("_R") for l in labels])
ALL = np.array(list(combinations(range(72), 4)))

CASES = [("S1 delivery, beta (both controls)", np.asarray(s1["beta_both_d"])),
         ("S2 attitude, alpha (+ features)",    np.asarray(m23["S2_alpha_feat"])),
         ("S2 attitude, theta (+ features)",    np.asarray(m23["S2_theta_feat"])),
         ("S2 attitude, beta  (+ features)",    np.asarray(m23["S2_beta_feat"])),
         ("S3 cue, alpha  (negative control)",  np.asarray(m23["S3_alpha_feat"])),
         ("S3 cue, beta   (negative control)",  np.asarray(m23["S3_beta_feat"]))]


def adjacency(thr):
    A = np.zeros((72, 72), bool)
    for i, j in combinations(range(72), 2):
        if cdist(parcel_pts[i], parcel_pts[j]).min() <= thr: A[i, j] = A[j, i] = True
    return A


def connected(s, A):
    """True if the parcels in `s` form one connected component under A."""
    s = list(s); seen = {s[0]}; stack = [s[0]]
    while stack:
        u = stack.pop()
        for v in s:
            if v not in seen and A[u, v]: seen.add(v); stack.append(v)
    return len(seen) == len(s)


def cell(v, sets, obs):
    """Exceedance count and exact p over a complete reference set."""
    m = v[sets].mean(1); k = int((m >= obs).sum()); n = len(m)
    return k, n, (1 + k) / (n + 1), m.std()


cache = {}
for thr in THRESHOLDS:
    A = adjacency(thr)
    if not connected(roi, A):
        print(f"threshold {thr} cm: the ROI is NOT connected -- skipping"); continue
    conn = np.array([connected(c, A) for c in ALL])
    CONN = ALL[conn]; CONN_R = ALL[conn & right[ALL].all(1)]
    cache[thr] = A
    print(f"\n===== adjacency threshold {thr} cm "
          f"(mean degree {A.sum(1).mean():.1f}, range {A.sum(1).min()}-{A.sum(1).max()}) =====")
    print(f"reference sets: all={len(ALL)}  connected={len(CONN)}  connected & right={len(CONN_R)}")
    print(f"{'contrast':36s} {'mean d':>8s} | {'connected':>22s} | {'connected + right':>22s}")
    for name, v in CASES:
        obs = v[roi].mean()
        kc, nc, pc, sc = cell(v, CONN, obs)
        kr, nr, pr, _ = cell(v, CONN_R, obs)
        print(f"{name:36s} {obs:+8.4f} | {kc:6d}/{nc:<6d} p={pc:7.4f} | {kr:6d}/{nr:<6d} p={pr:7.4f}")
    ref = np.asarray(m23["S2_alpha_feat"])                 # widening of the null, on one fixed map
    _, _, _, sd_all = cell(ref, ALL, ref[roi].mean())
    _, _, _, sd_conn = cell(ref, CONN, ref[roi].mean())
    print(f"  (null SD for the S2 alpha map: {sd_all:.4f} over all sets, {sd_conn:.4f} over "
          f"connected sets -- the widening the restriction is meant to capture)")

np.savez_compressed(os.path.join(DERIV, "roi_adjacency.npz"),
                    labels=np.array(labels, object), dim=np.array(DIM),
                    **{f"adjacency_{str(t).replace('.', 'p')}cm": cache[t] for t in cache})
print(f"\nadjacency matrices written to {os.path.join(DERIV, 'roi_adjacency.npz')}")
