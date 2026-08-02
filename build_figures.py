"""
Assemble the numbered figure set for submission.

    Figure 1  experimental designs (drawn here)
    Figure 2  (A) power spectrum + specparam        (B) global ISC, Study 1
    Figure 3  Study 1 beta ISC, raw and surviving core
    Figure 4  (A) Study 2 attitude alpha/theta maps (B) global ISC, Study 2
    Figure 5  synchronisation as a trait
    Figure 6  cross-study convergence: (A) maps with the ROI outlined (B) enrichment nulls

Figures 2 and 4 are two panels each, produced by separate scripts and composited here; 3, 5 and 6
are single images that only need re-stamping at print resolution. Panel letters are drawn into a
left gutter rather than into the panels themselves, so no panel is cropped or overwritten.

Wiley asks for at least 300 dpi, a width between 80 and 180 mm, under 10 MB, and files named by
figure number alone. Anything wider than 180 mm at 300 dpi is downsampled to fit; the printed size
of each output is reported so that a figure that has become too small to read is visible before
submission rather than after.

Run the panel scripts first:
    fig_power_spectrum.py, fig_global_isc.py                        -> Figure 2
    fig_export_render_stats.py -> matlab/render_patch.m -> fig_brain_montage.py -> Figures 3, 4A
    fig_global_isc_s23.py                                           -> Figure 4B
    fig_trait_synchrony.py                                          -> Figure 5
    fig_convergence_export.py -> matlab/render_convergence.m -> fig_convergence.py -> Figure 6

Requires Pillow in addition to matplotlib.

Usage: python build_figures.py
"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image, ImageDraw, ImageFont

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"G:/Barak1"
DERIV = os.path.join(ROOT, "reanalysis")   # reconstruction outputs and intermediate results; point anywhere

OUT = os.path.join(DERIV, "figures_final")
os.makedirs(OUT, exist_ok=True)
DPI = 300
MAXW = int(180 / 25.4 * DPI)               # 180 mm at 300 dpi, the Wiley maximum


# ---------------------------------------------------------------- Figure 1: the designs
def figure1():
    fig, axes = plt.subplots(2, 1, figsize=(6.0, 4.2))
    C_CH, C_NC, C_SI = "#c0392b", "#2980b9", "#7f8c8d"
    C_POS, C_NEG, C_CUE, C_NOCUE = "#c0392b", "#2980b9", "#16a085", "#95a5a6"

    def box(ax, x, w, y, h, fc, label, sub=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.02",
                                    fc=fc, ec="none", alpha=.85))
        ax.text(x + w/2, y + h/2 + (.055 if sub else 0), label, ha="center", va="center",
                fontsize=8, color="w", fontweight="bold")
        if sub:
            ax.text(x + w/2, y + h/2 - .085, sub, ha="center", va="center", fontsize=6.5, color="w")

    ax = axes[0]
    ax.set_title("Study 1  -  Speaker delivery (within-subjects, N = 40)",
                 fontsize=8.5, fontweight="bold", loc="left", pad=5)
    box(ax, .02, .29, .52, .34, C_CH, "Charismatic", "same speech, same speaker")
    box(ax, .345, .29, .52, .34, C_NC, "Non-charismatic", "same speech, same speaker")
    box(ax, .67, .29, .52, .34, C_SI, "Silent video", "control")
    ax.annotate("", xy=(.97, .42), xytext=(.02, .42),
                arrowprops=dict(arrowstyle="-|>", color="0.35", lw=1.0))
    ax.text(.5, .21, "counterbalanced order  |  MEG throughout", ha="center", fontsize=6.5, color="0.3")
    ax.text(.5, .04, "contrast: charismatic vs non-charismatic ISC", ha="center", fontsize=7,
            style="italic", color="0.15")

    ax = axes[1]
    ax.set_title("Study 2  -  Listener attitude x social context (one cohort, two crossed manipulations)",
                 fontsize=8.5, fontweight="bold", loc="left", pad=5)
    ax.text(.235, .80, "Manipulation 1: ATTITUDE  (n = 58)", ha="center", fontsize=7,
            fontweight="bold", color="0.2")
    box(ax, .015, .20, .38, .30, C_POS, "Positive", "n = 29")
    box(ax, .255, .20, .38, .30, C_NEG, "Negative", "n = 29")
    ax.text(.235, .29, "identical lecture, differing prior description", ha="center",
            fontsize=6.2, color="0.35")
    ax.text(.765, .80, "Manipulation 2: AUDIENCE CUE  (n = 61)", ha="center", fontsize=7,
            fontweight="bold", color="0.2")
    box(ax, .545, .20, .38, .30, C_CUE, "Cue present", "n = 28")
    box(ax, .785, .20, .38, .30, C_NOCUE, "No cue", "n = 33")
    ax.text(.765, .29, "second, different lecture; silhouetted audience", ha="center",
            fontsize=6.2, color="0.35")
    ax.plot([.50, .50], [.15, .88], color="0.75", lw=.8, ls="--")
    ax.text(.5, .04, "65 recruited; both tasks in one session, counterbalanced (overlap n = 58)",
            ha="center", fontsize=7, style="italic", color="0.15")

    for ax in axes:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.tight_layout(h_pad=1.6)
    p = os.path.join(OUT, "Figure1.png")
    fig.savefig(p, dpi=DPI, facecolor="w"); plt.close(fig)
    return p


# ---------------------------------------------------------------- compositing
def _cap(im):
    """Downsample to the 180 mm print-width limit if necessary."""
    if im.width > MAXW:
        im = im.resize((MAXW, round(im.height * MAXW / im.width)), Image.LANCZOS)
    return im


def _font(sz):
    for f in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try: return ImageFont.truetype(f, sz)
        except OSError: pass
    return ImageFont.load_default()


def stack(paths, out, letters=("A", "B"), gap=26, pad_left=54):
    """Stack panels vertically, centred, with panel letters in a left gutter."""
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print(f"  SKIPPED {os.path.basename(out)}: missing {[os.path.basename(m) for m in missing]}")
        return None
    ims = [Image.open(p).convert("RGB") for p in paths]
    W = max(i.width for i in ims) + pad_left
    H = sum(i.height for i in ims) + gap * (len(ims) - 1)
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas); f = _font(40)
    y = 0
    for im, L in zip(ims, letters):
        canvas.paste(im, (pad_left + (W - pad_left - im.width) // 2, y))
        d.text((10, y + 6), L, fill="black", font=f)
        y += im.height + gap
    _cap(canvas).save(out, dpi=(DPI, DPI))
    return out


def relabel(path, out):
    """Single-panel figure: re-stamp at print resolution."""
    if not os.path.exists(path):
        print(f"  SKIPPED {os.path.basename(out)}: missing {os.path.basename(path)}")
        return None
    _cap(Image.open(path).convert("RGB")).save(out, dpi=(DPI, DPI))
    return out


if __name__ == "__main__":
    S = lambda n: os.path.join(DERIV, n)
    O = lambda n: os.path.join(OUT, n)
    made = [figure1()]
    made.append(stack([S("fig_power_spectrum.png"), S("fig_global_isc.png")], O("Figure2.png")))
    # fig_brain_study1_beta.png, not the older fig_brain_study1_raw_vs_core.png: the current
    # version masks the lower panel by the surviving-stage significance rather than the raw
    made.append(relabel(S("fig_brain_study1_beta.png"), O("Figure3.png")))
    made.append(stack([S("fig_brain_study2.png"), S("fig_global_isc_s23.png")], O("Figure4.png")))
    made.append(relabel(S("fig_trait_synchrony.png"), O("Figure5.png")))
    made.append(relabel(S("fig_convergence.png"), O("Figure6.png")))

    print(f"\n{'file':16s} {'pixels':>14s} {'width @300dpi':>15s} {'MB':>6s}")
    for p in [m for m in made if m]:
        w, h = Image.open(p).size
        mm = w / DPI * 25.4
        flag = "  <- under 80 mm" if mm < 80 else ""
        print(f"{os.path.basename(p):16s} {w:6d} x {h:<5d} {mm:12.1f} mm {os.path.getsize(p)/1e6:6.2f}{flag}")
