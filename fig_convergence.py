"""
Figure 6: cross-study convergence.

Panel A -- the three effect maps on a common colour scale, with the a priori ROI outlined:
Study 1 delivery (beta, after both controls), Study 2 attitude (alpha, after feature control),
Study 2 audience cue (alpha, after feature control). Rendered by matlab/render_convergence.m
from fig_convergence_export.py.

Panel B -- what the enrichment test actually does. For each contrast, the null distribution of
the mean effect over all 1,028,790 four-parcel sets, with the observed ROI mean marked. The two
message-relevant contrasts sit in the far right tail; the cue sits in the body of its own null,
which is what a null result looks like and why its p carries no information about location.

Requires: <DERIV>/renders/conv_*.png, study1_source_myica3_greater_pc0.npz,
<DERIV>/s23_stage_maps.npz.

Usage: python fig_convergence.py
"""
import os, sys
from itertools import combinations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"G:/Barak1"
DERIV = os.path.join(ROOT, "reanalysis")   # reconstruction outputs and intermediate results; point anywhere

RENDERS = os.path.join(DERIV, "renders")
OUT = os.path.join(DERIV, "fig_convergence.png")
ROI = ["Rolandic_Oper_R", "Temporal_Sup_R", "Temporal_Mid_R", "Insula_R"]

s1 = np.load(os.path.join(ROOT, "paper_code", "study1_source_myica3_greater_pc0.npz"))
m23 = np.load(os.path.join(DERIV, "s23_stage_maps.npz"), allow_pickle=True)
labels = [str(x).strip() for x in s1["labels"]]
roi = [labels.index(r) for r in ROI]

CONTRASTS = [
    ("conv_s1_delivery", "Study 1  delivery", "beta, after feature + aperiodic control",
     np.asarray(s1["beta_both_d"]), "#1f4e9c"),
    ("conv_s2_attitude", "Study 2  attitude", "alpha, after feature control",
     np.asarray(m23["S2_alpha_feat"]), "#0b7a5a"),
    ("conv_s3_cue", "Study 2  audience cue", "alpha, after feature control",
     np.asarray(m23["S3_alpha_feat"]), "#9b7a1a"),
]

# the null: mean effect over every four-parcel set (the enrichment test's reference set)
ALL = np.array(list(combinations(range(72), 4)), dtype=np.int8)
print(f"reference set: {len(ALL):,} four-parcel sets")

fig = plt.figure(figsize=(11.0, 8.4))
gs = GridSpec(2, 3, height_ratios=[1.25, 1.0], hspace=0.28, wspace=0.16,
              left=0.055, right=0.985, top=0.925, bottom=0.155)

# ---------------------------------------------------------------- Panel A
for j, (key, name, stage, _, _) in enumerate(CONTRASTS):
    ax = fig.add_subplot(gs[0, j])
    p = os.path.join(RENDERS, f"{key}_Rlat.png")
    if os.path.exists(p):
        im = mpimg.imread(p)
        h, w = im.shape[:2]                        # trim the surrounding whitespace
        ax.imshow(im[int(0.14*h):int(0.86*h), int(0.10*w):int(0.90*w)])
    else:
        ax.text(0.5, 0.5, f"missing\n{key}_Rlat.png", ha="center", va="center",
                fontsize=8, color="#999999", transform=ax.transAxes)
    ax.axis("off")
    ax.set_title(name, fontsize=11, fontweight="bold", pad=2)
    ax.text(0.5, -0.04, stage, fontsize=8.5, color="#444444",
            ha="center", va="top", transform=ax.transAxes)

# same scale as Figures 3 and 4: sequential, zero-based, negatives clipped (all tests one-sided)
CLIM = 0.09
cax = fig.add_axes([0.36, 0.545, 0.28, 0.016])
grad = np.linspace(0, CLIM, 256)[None, :]
cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "grey_red", [(0.78, 0.78, 0.78), (0.93, 0.45, 0.32), (0.55, 0.0, 0.0)])
cax.imshow(grad, aspect="auto", cmap=cmap, extent=[0, CLIM, 0, 1])
cax.set_yticks([]); cax.set_xticks([0, CLIM / 2, CLIM])
cax.set_xticklabels(["0", f"{CLIM/2:.03f}", f"{CLIM:.02f}"], fontsize=8)
cax.set_xlabel("ISC difference (Fisher $z$)", fontsize=8.5, labelpad=2)
for s in cax.spines.values(): s.set_linewidth(0.5)

# ---------------------------------------------------------------- Panel B
# One x-axis for all three, so the widths of the nulls are comparable as well as the positions
# of the observed values: the cue's null is genuinely narrower, and per-panel limits hide that.
nulls = {key: vals[ALL].mean(1) for key, _, _, vals, _ in CONTRASTS}
lo = min(np.percentile(v, 0.05) for v in nulls.values())
hi = max(max(np.percentile(v, 99.95) for v in nulls.values()),
         max(vals[roi].mean() for _, _, _, vals, _ in CONTRASTS)) * 1.10
bins = np.linspace(lo, hi, 150)

for j, (key, name, stage, vals, colour) in enumerate(CONTRASTS):
    ax = fig.add_subplot(gs[1, j])
    null = nulls[key]
    obs = vals[roi].mean()
    k = int((null >= obs).sum()); p = (1 + k) / (len(null) + 1)
    ax.hist(null, bins=bins, color="#c9c9c9", edgecolor="none")
    ax.axvline(obs, color=colour, lw=2.2)
    ax.set_xlim(lo, hi)
    ax.set_yticks([])
    ax.set_xlabel("mean effect in a four-parcel set", fontsize=8.5)
    if j == 0: ax.set_ylabel("four-parcel sets", fontsize=9)
    ptxt = f"$p$ = {p:.5f}" if p >= 1e-4 else f"$p$ = {p:.6f}"
    # label below the axis rather than inside it, so nothing overlies the distribution
    ax.text(0.5, -0.30, f"observed ROI {obs:+.3f},   {ptxt}\n{k:,} of {len(null):,} sets exceed",
            transform=ax.transAxes, fontsize=8.5, color=colour, va="top", ha="center")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    print(f"{name:24s} obs={obs:+.4f}  {k:>7,}/{len(null):,}  p={p:.6f}")

fig.text(0.012, 0.965, "A", fontsize=15, fontweight="bold")
fig.text(0.012, 0.470, "B", fontsize=15, fontweight="bold")
fig.savefig(OUT, dpi=220)
print(f"\nsaved {OUT}")
