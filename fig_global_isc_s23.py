"""
Figure 4B: global inter-subject correlation by band and group, both Study 2 manipulations.

Left, the attitude contrast; right, the audience cue. Bars are the mean across participants of the
72-region-average leave-one-out ISC, computed within group, with the standard error.

SIGNIFICANCE. Bands are starred from the group-label permutation test of Section 2.5 -- the same
test the Results report -- and only where the manipulation raises correlation, which is the
predicted direction. An independent-samples t-test on the leave-one-out values held fixed is not
equivalent and is anti-conservative here: those values are computed against a group mean each
participant contributes to, so they are not independent observations. On the attitude contrast the
t-test puts delta at p = .015 and the permutation test at p = .06, so the parametric version stars
a band the permutation test does not support.

The permutation adds about a minute on top of the load, so the test is run here rather than read
from elsewhere: the asterisks and the reported p values then come from one computation.

Per-subject values are cached to <DERIV>/global_isc_s23.npz; delete it to recompute.

Requires: source_iscex/ from matlab/source_reconstruction_iscex.m.

Usage: python fig_global_isc_s23.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lib2 import load, loo, perm_global, BANDS, NPERM

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"G:/Barak1"
DERIV = os.path.join(ROOT, "reanalysis")   # reconstruction outputs and intermediate results; point anywhere

CACHE = os.path.join(DERIV, "global_isc_s23.npz")
OUT = os.path.join(DERIV, "fig_global_isc_s23.png")
RNG = np.random.RandomState(41)

PANELS = [
    ("S2", "S2_band", "include_S2", "attitude_grp", "Positive", "Negative",
     ("#2c6fbb", "#c0392b"), "Listener attitude"),
    ("S3", "S3_band", "include_S3", "cue_grp", "With", "Without",
     ("#27ae60", "#95a5a6"), "Audience cue"),
]

if os.path.exists(CACHE):
    c = dict(np.load(CACHE, allow_pickle=True))
    print(f"loaded {CACHE}")
else:
    c = {}
    for tag, field, inc, gcol, A, B, _, _ in PANELS:
        X, grp, _ = load(field, inc, gcol)
        gA = np.where(grp == A)[0]; gB = np.where(grp == B)[0]
        G = np.zeros((X.shape[0], 5))
        for bi in range(5):
            z = np.zeros((X.shape[0], 72))
            for gi in (gA, gB): z[gi] = loo(X[gi, :, :, bi])      # leave-one-out WITHIN group
            G[:, bi] = z.mean(1)
        p = np.array([perm_global(X[:, :, :, bi], gA, gB, rng=RNG)[1] for bi in range(5)])
        c[f"{tag}_G"] = G; c[f"{tag}_grp"] = grp; c[f"{tag}_p"] = p
        print(f"{tag}: n={X.shape[0]} ({A}={len(gA)}, {B}={len(gB)}), "
              f"permutation p = {np.array2string(p, precision=3)}", flush=True)
        del X
    np.savez(CACHE, **c)
    print(f"cached to {CACHE}")

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)
x = np.arange(5); w = 0.36

for ax, (tag, _, _, _, A, B, cols, title) in zip(axes, PANELS):
    G = c[f"{tag}_G"]; grp = np.asarray(c[f"{tag}_grp"], dtype=str); p = c[f"{tag}_p"]
    for j, (gl, col) in enumerate([(A, cols[0]), (B, cols[1])]):
        sub = G[grp == gl]
        ax.bar(x + (j - 0.5) * w, sub.mean(0), w, yerr=sub.std(0) / np.sqrt(len(sub)),
               capsize=2, color=col, label=f"{gl} (n = {len(sub)})", error_kw=dict(lw=0.8))
    for bi in range(5):
        e = G[grp == A][:, bi]; k = G[grp == B][:, bi]
        if p[bi] < .05 and e.mean() > k.mean():            # predicted direction only
            ax.text(x[bi], max(e.mean(), k.mean()) + 0.010, "*", ha="center", fontsize=14)
    ax.set_xticks(x); ax.set_xticklabels(BANDS)
    ax.set_title(title, fontsize=10.5)
    ax.legend(fontsize=8, frameon=False)
    ax.axhline(0, color="k", lw=.5)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axes[0].set_ylabel("Global ISC (Fisher $z$)")

fig.text(0.5, 0.005, f"* p < .05, group-label permutation ({NPERM:,} permutations), "
                     "one-sided in the predicted direction",
         ha="center", fontsize=8, color="#444444")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(OUT, dpi=160)
print(f"\nsaved {OUT}")
for tag, _, _, _, A, B, _, title in PANELS:
    G = c[f"{tag}_G"]; grp = np.asarray(c[f"{tag}_grp"], dtype=str); p = c[f"{tag}_p"]
    print(f"\n{title}: {A} / {B}, permutation p")
    for bi, b in enumerate(BANDS):
        star = " *" if p[bi] < .05 and G[grp == A][:, bi].mean() > G[grp == B][:, bi].mean() else ""
        print(f"  {b:6s} {G[grp==A][:,bi].mean():.3f} / {G[grp==B][:,bi].mean():.3f}   p = {p[bi]:.4f}{star}")
