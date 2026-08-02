"""
Figure 5: synchronisation as a stable individual trait.

Left, each participant's alpha ISC on the attitude task against their alpha ISC on the audience-cue
task -- two different speakers, two different lectures, the same listener. Right, the same
cross-task correlation in every band, which is what makes the alpha result specific rather than a
general signal-quality effect: if the relationship were driven by how clean a participant's data
were, it would appear in delta and gamma too.

Every value plotted is read from <DERIV>/trait_synchrony.npz, written by
study2_trait_synchrony.py. Nothing here is transcribed, so the figure cannot drift away from the
numbers the Results report.

Requires: study2_trait_synchrony.py to have been run.

Usage: python fig_trait_synchrony.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"G:/Barak1"
DERIV = os.path.join(ROOT, "reanalysis")   # reconstruction outputs and intermediate results; point anywhere

SRC = os.path.join(DERIV, "trait_synchrony.npz")
OUT = os.path.join(DERIV, "fig_trait_synchrony.png")

d = np.load(SRC, allow_pickle=True)
x, y = d["alpha_x"], d["alpha_y"]
rel_r, rel_p = d["rel_r"], d["rel_p"]
bands = [str(b) for b in d["bands"]]
r, p, sp, r_nox, n = float(d["r"]), float(d["p"]), float(d["spearman"]), float(d["r_drop_extreme"]), int(d["n"])

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw={"width_ratios": [1.25, 1]})

# ---------------------------------------------------------------- left: the scatter
ax[0].scatter(x, y, s=42, c="#2c6fbb", edgecolor="w", linewidth=0.6, alpha=0.9, zorder=3)
b1, b0 = np.polyfit(x, y, 1)
xs = np.array([x.min(), x.max()])
ax[0].plot(xs, b0 + b1 * xs, "#c0392b", lw=2, zorder=2)
ax[0].set_xlabel("alpha ISC, attitude task")
ax[0].set_ylabel("alpha ISC, audience-cue task")
ax[0].set_title("Synchronisation is stable across speakers", fontsize=12)
ptxt = f"{p:.0e}" if p < .001 else f"{p:.3f}"
ax[0].text(0.03, 0.97,
           f"Pearson $r$ = {r:+.2f} ($p$ = {ptxt}, $n$ = {n})\n"
           f"Spearman $r$ = {sp:+.2f};  without the extreme point, $r$ = {r_nox:+.2f}",
           transform=ax[0].transAxes, va="top", ha="left", fontsize=9.5,
           bbox=dict(boxstyle="round", fc="#f2f6fb", ec="#bcccdc"))

# ---------------------------------------------------------------- right: band specificity
cols = ["#2c6fbb" if pp < .05 else "#b8c4d0" for pp in rel_p]
ax[1].bar(range(len(bands)), rel_r, color=cols, edgecolor="none")
ax[1].axhline(0, color="k", lw=.6)
ax[1].set_xticks(range(len(bands))); ax[1].set_xticklabels(bands)
ax[1].set_ylabel("cross-task ISC correlation ($r$)")
ax[1].set_title("Specific to alpha and beta, not delta or gamma", fontsize=12)
for i, pp in enumerate(rel_p):
    if pp < .05: ax[1].text(i, rel_r[i] + .02, "*", ha="center", fontsize=15)
ax[1].set_ylim(0, max(rel_r) * 1.18)          # headroom so the tallest star clears the title
for s in ("top", "right"): ax[1].spines[s].set_visible(False)

fig.tight_layout()
fig.savefig(OUT, dpi=160)
print(f"saved {OUT}")
print(f"  alpha: r = {r:+.2f}, p = {p:.2e}, n = {n}")
for b, rr, pp in zip(bands, rel_r, rel_p):
    print(f"  {b:6s} r = {rr:+.2f}  p = {pp:.4f}{'  *' if pp < .05 else ''}")
