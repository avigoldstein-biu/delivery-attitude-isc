"""
Analysis A (visual): low-level visual features of the Charismatic vs Non-Charismatic
clips -- motion energy, luminance, RMS contrast -- decoded via ffmpeg. Addresses the
concern that basic visual dynamics could differ between conditions and
contribute to ISC (esp. in occipital regions, where the paper found strong effects).
"""
import os, json, subprocess, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import imageio_ffmpeg
ROOT = r"G:/Barak1"
DERIV = os.path.join(ROOT, "reanalysis")   # reconstruction outputs and intermediate results; point anywhere
OUT = DERIV

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
W, H, FPS = 160, 128, 25
CLIPS = {"Charismatic": "Charismatic.MPG", "Non-Charismatic": "Non-Charismatic.MPG"}

def decode_gray(mpg):
    cmd = [FFMPEG, "-loglevel", "error", "-i", os.path.join(ROOT, mpg),
           "-vf", f"scale={W}:{H},format=gray,fps={FPS}", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    n = len(raw) // (W*H)
    return np.frombuffer(raw[:n*W*H], np.uint8).reshape(n, H, W).astype(float)/255.0

def vis_features(F):
    lum = F.mean((1, 2))                                   # luminance per frame
    contrast = F.std((1, 2))                               # spatial RMS contrast
    motion = np.abs(np.diff(F, axis=0)).mean((1, 2))       # frame-to-frame motion energy
    motion = np.concatenate([[motion[0]], motion])
    return dict(lum=lum, contrast=contrast, motion=motion)

res, curves = {}, {}
for name, mpg in CLIPS.items():
    F = decode_gray(mpg); v = vis_features(F); curves[name] = v
    res[name] = {"n_frames": len(F), "dur_s": round(len(F)/FPS, 1),
                 "lum_mean": float(v["lum"].mean()), "lum_cv": float(v["lum"].std()/v["lum"].mean()),
                 "contrast_mean": float(v["contrast"].mean()),
                 "motion_mean": float(v["motion"].mean()), "motion_cv": float(v["motion"].std()/v["motion"].mean())}
    print(f"[{name}] {len(F)} frames")

print("\n%-16s %12s %12s %10s" % ("feature", "Charismatic", "Non-Charis", "ratio C/NC"))
for k in ["lum_mean", "lum_cv", "contrast_mean", "motion_mean", "motion_cv"]:
    c, n = res["Charismatic"][k], res["Non-Charismatic"][k]
    print("%-16s %12.4f %12.4f %10.2f" % (k, c, n, c/n if n else np.nan))
json.dump(res, open(os.path.join(OUT, "visual_features.json"), "w"), indent=2)

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for name in CLIPS:
    t = np.arange(len(curves[name]["motion"]))/FPS
    ax[0].plot(t, curves[name]["motion"], label=name, alpha=.7, lw=.6)
ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("motion energy"); ax[0].legend(); ax[0].set_title("Motion energy over time")
ax[1].bar(["lum", "contrast", "motion"],
          [res["Charismatic"]["lum_mean"], res["Charismatic"]["contrast_mean"], res["Charismatic"]["motion_mean"]],
          width=-0.4, align="edge", label="Charismatic")
ax[1].bar(["lum", "contrast", "motion"],
          [res["Non-Charismatic"]["lum_mean"], res["Non-Charismatic"]["contrast_mean"], res["Non-Charismatic"]["motion_mean"]],
          width=0.4, align="edge", label="Non-Charismatic")
ax[1].set_ylabel("mean value"); ax[1].legend(); ax[1].set_title("Visual feature means")
plt.tight_layout(); fig.savefig(os.path.join(OUT, "visual_features.png"), dpi=110)
print("\nSaved visual_features.json + visual_features.png")
