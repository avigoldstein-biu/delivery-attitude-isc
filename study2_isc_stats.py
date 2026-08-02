"""
01 - Inter-subject correlation for Study 2, tested at the SUBJECT level.

Study 2 crossed two manipulations in one cohort: ATTITUDE (a positive or negative description of
the speaker read before an identical lecture; field S2_band) and the AUDIENCE CUE (silhouettes
present or absent during a second, different lecture; field S3_band). Each participant has one
level of each factor, so leave-one-out ISC is computed WITHIN group and compared BETWEEN groups by
permuting the group labels. Under permutation the leave-one-out ISC is RE-COMPUTED within the
permuted groups, via the per-region ISC Gram matrix -- permuting frozen subject-level values
instead treats them as exchangeable when they are not, and returns p values several times too
small.

Reports, per band:
  - global ISC (the 72-region mean) for the group difference, with the within-cohort baseline
  - the per-region difference with both the two-sided and the one-sided (effect > control)
    permutation p, and FDR across the 72 regions under each

Both sidedness conventions are reported because Section 4.3 uses both: three right-perisylvian
regions survive two-sided FDR in alpha, and the superior temporal gyrus, the orbital superior
frontal gyrus and the left paracentral lobule survive only under the a priori direction. Section
2.5 fixes the permutation count at 10,000 (lib2.NPERM).

The global test uses the same permutation as the regional one. An independent-samples t-test on
the leave-one-out values, holding them fixed, is not equivalent: those values are computed against
a group mean that each participant contributes to, so they are not independent observations, and
the t-test returns p values several times too small. The difference is not academic here -- it
moves the attitude contrast in theta from p < .001 to .004, in alpha from .0001 to .012, and in
delta from .015 to .062.

Writes <DERIV>/study2_isc_results.npz, which is read by cross_study_convergence.py,
roi_contiguity_null.py and fig_export_render_stats.py.

Requires: source_iscex/ from matlab/source_reconstruction_iscex.m.

Usage: python study2_isc_stats.py
"""
import gc, os, sys
import numpy as np
from lib2 import load, perm_between, perm_global, fdr, labels, BANDS, NPERM

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"G:/Barak1"
DERIV = os.path.join(ROOT, "reanalysis")   # reconstruction outputs and intermediate results; point anywhere

OUT = os.path.join(DERIV, "study2_isc_results.npz")

# One stream for the regional test, consumed study by study and band by band, so the reported p
# values are reproducible from the seed alone. The global test draws from its own stream, so adding
# or removing it cannot move the regional numbers.
RNG_REGION = np.random.RandomState(0)
RNG_GLOBAL = np.random.RandomState(1)

# Result keys stay S2_*/S3_*: they name the MATLAB fields the two tasks are stored in
# (S2_band, S3_band). Both belong to Study 2 of the manuscript, which is how they are labelled in
# everything a reader sees.
CONTRASTS = [
    ("S2", "S2_band", "include_S2", "attitude_grp", "Positive", "Negative",
     "Study 2: listener attitude (positive - negative)"),
    ("S3", "S3_band", "include_S3", "cue_grp", "With", "Without",
     "Study 2: audience cue (cue - no cue)"),
]

out = {"labels": np.array(labels)}

# The two region time-course sets are several gigabytes each, so only one is held at a time and
# the run is dominated by reading them rather than by the permutations.
for tag, field, inc, gcol, A, B, title in CONTRASTS:
    X = None; gc.collect()
    X, grp, _ = load(field, inc, gcol)
    gA = np.where(grp == A)[0]; gB = np.where(grp == B)[0]
    print(f"\n===== {title} =====  n={X.shape[0]} ({A}={len(gA)}, {B}={len(gB)}), "
          f"T={X.shape[2]}, {NPERM:,} permutations")

    print("\n  global ISC, 72-region mean (one-sided, effect > control):")
    for bi, b in enumerate(BANDS):
        d, p, base = perm_global(X[:, :, :, bi], gA, gB, rng=RNG_GLOBAL)
        print(f"    {b:6s} d={d:+.4f} p={p:.4f}{'  *' if p < .05 else ''}   (baseline ISC={base:.3f})")

    print("\n  regional (FDR across 72 regions):")
    for bi, b in enumerate(BANDS):
        obs, p2, p1 = perm_between(X[:, :, :, bi], gA, gB, rng=RNG_REGION)
        sig2, sig1 = fdr(p2), fdr(p1)
        out[f"{tag}_{b}_obs"] = obs; out[f"{tag}_{b}_p2"] = p2; out[f"{tag}_{b}_p1"] = p1
        print(f"    {b:6s}: two-sided {sig2.sum():2d} sig | one-sided {sig1.sum():2d} sig")
        for r in np.argsort(-obs):
            if sig2[r] or sig1[r]:
                print(f"        {labels[r]:22s} d={obs[r]:+.4f} p2={p2[r]:.4f} p1={p1[r]:.4f}"
                      f"{'  [2s]' if sig2[r] else ''}{'  [1s]' if sig1[r] else ''}")
    del X

np.savez(OUT, **out)
print(f"\nsaved {OUT}")
