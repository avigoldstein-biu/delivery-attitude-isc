"""
Assemble the numbered figure set for submission.

    Figure 2  (A) power spectrum + specparam        (B) global ISC, Study 1
    Figure 3  Study 1 beta ISC, raw and surviving core
    Figure 4  (A) Study 2 attitude alpha/theta maps (B) global ISC, Study 2
    Figure 5  synchronisation as a trait
    Figure 6  cross-study convergence: (A) maps with the ROI outlined (B) enrichment nulls

Figures 2 and 4 are two panels each, produced by separate scripts and composited here; 3, 5 and 6
are single images that only need re-stamping at print resolution. Panel letters are drawn into a
left gutter rather than into the panels themselves, so no panel is cropped or overwritten.

Figure 1 is not built here. It is a schematic of the two designs, drawn by hand as SVG rather than
generated from data, and it is submitted as `figures_final/Figure1.svg` converted to PDF. Nothing
in this repository reproduces it, and it is left out rather than regenerated so that running this
script cannot overwrite the version that goes to the journal.

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

Requires Pillow.

Usage: python build_figures.py
"""
import os, sys
from PIL import Image, ImageDraw, ImageFont

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"G:/Barak1"
DERIV = os.path.join(ROOT, "reanalysis")   # reconstruction outputs and intermediate results; point anywhere

OUT = os.path.join(DERIV, "figures_final")
os.makedirs(OUT, exist_ok=True)
DPI = 300
MAXW = int(180 / 25.4 * DPI)               # 180 mm at 300 dpi, the Wiley maximum


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
    made = []
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
