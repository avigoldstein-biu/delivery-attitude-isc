"""Composite the per-view brain renders into paper-style montages with labels + colorbar."""
import os
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap
ROOT = r"G:/Barak1"
DERIV = os.path.join(ROOT, "reanalysis")   # reconstruction outputs and intermediate results; point anywhere
R = os.path.join(DERIV, "renders/")
CLIM = (0, 0.09)
# control points shared with matlab/render_patch.m and fig_convergence.py
GREY_RED = LinearSegmentedColormap.from_list(
    "grey_red", [(0.78, 0.78, 0.78), (0.93, 0.45, 0.32), (0.55, 0.00, 0.00)])

def autocrop(path, pad=6):
    im = Image.open(path).convert("RGB"); a = np.asarray(im)
    nonwhite = (a < 245).any(2)
    if not nonwhite.any(): return a
    ys, xs = np.where(nonwhite)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    y0 = max(0, y0-pad); x0 = max(0, x0-pad); y1 = min(a.shape[0], y1+pad); x1 = min(a.shape[1], x1+pad)
    return a[y0:y1, x0:x1]

COLS = [("Llat", "L lateral"), ("Lmed", "L medial"), ("Rmed", "R medial"), ("Rlat", "R lateral")]

def montage(panels, rowlabels, title, out):
    nr, nc = len(panels), len(COLS)
    fig, axes = plt.subplots(nr, nc, figsize=(2.6*nc, 2.5*nr+0.6), squeeze=False)
    for ri, panel in enumerate(panels):
        for ci, (vk, vlab) in enumerate(COLS):
            ax = axes[ri][ci]; ax.axis("off")
            ax.imshow(autocrop(f"{R}{panel}_{vk}.png"))
            if ri == 0: ax.set_title(vlab, fontsize=11)
        axes[ri][0].text(-0.08, 0.5, rowlabels[ri], transform=axes[ri][0].transAxes,
                         rotation=90, va="center", ha="center", fontsize=12, fontweight="bold")
    fig.suptitle(title, fontsize=13, y=0.99)
    # Same grey-to-red ramp as matlab/render_patch.m and fig_convergence.py: the colourbar has to
    # match the surfaces it labels, and one quantity should have one scale across the figure set.
    sm = ScalarMappable(norm=Normalize(*CLIM), cmap=GREY_RED)
    cax = fig.add_axes([0.25, 0.045, 0.5, 0.02])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("ISC difference (Fisher-z)", fontsize=10)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.10, wspace=0.02, hspace=0.05)
    fig.savefig(out, dpi=150); print("saved", out)

# Figure 3: the full control progression. 
montage(["s1_beta_raw", "s1_beta_core"],
        ["raw", "after feature +\naperiodic removal"],
        "Study 1: charismatic > non-charismatic beta ISC, before and after low-level control\n"
        "(only regions surviving FDR correction at that stage are shown)",
        os.path.join(DERIV, "fig_brain_study1_beta.png"))
montage(["s2_alpha", "s2_theta"], ["alpha", "theta"],
        "Study 2: positive > negative attitude ISC (right-perisylvian network)",
        os.path.join(DERIV, "fig_brain_study2.png"))
