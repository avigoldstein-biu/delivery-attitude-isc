"""
Recompute the Study 1 clip acoustic features after matching the two clips for level.

The .MPG source files differ ~3x in RMS, but playback was loudness-normalised, so the
level-dependent features computed from the source files. This rescales each clip to unit RMS before extraction so
the reported values reflect the presented stimulus.

Gain-invariant features (f0, coefficients of variation, spectral centroid, modulation-power
proportions) are unchanged by construction; they are recomputed anyway as a check.
"""
import os, json, subprocess, numpy as np, soundfile as sf, librosa
from scipy.signal import hilbert, butter, filtfilt, welch
import imageio_ffmpeg
ROOT = r"G:/Barak1"
DERIV = os.path.join(ROOT, "reanalysis")   # reconstruction outputs and intermediate results; point anywhere
OUT = DERIV

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
SR = 16000
CLIPS = {"Charismatic": "Charismatic.MPG", "Non-Charismatic": "Non-Charismatic.MPG"}


def load_audio(mpg, normalize):
    wav = os.path.join(OUT, "_tmpn.wav")
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", os.path.join(ROOT, mpg),
                    "-vn", "-ac", "1", "-ar", str(SR), "-f", "wav", wav], check=True)
    y, sr = sf.read(wav); os.remove(wav); y = y.astype(float)
    if normalize:
        y = y / np.sqrt((y**2).mean())          # unit RMS: matches presentation level
    return y, sr


def features(y, sr):
    f = {}
    e = np.abs(hilbert(y)); b, a = butter(4, 30/(sr/2), "low"); e = filtfilt(b, a, e)
    f["rms_mean"] = float(np.sqrt((y**2).mean()))
    f["env_mean"] = float(e.mean()); f["env_cv"] = float(e.std()/e.mean())
    d = np.diff(e, prepend=e[0]); d[d < 0] = 0
    f["onset_rate"] = float((d > d.mean()+2*d.std()).mean())
    f0, vflag, _ = librosa.pyin(y, fmin=75, fmax=350, sr=sr, frame_length=2048)
    v = f0[~np.isnan(f0)]
    f["voiced_frac"] = float(np.mean(vflag)); f["f0_median"] = float(np.median(v))
    f["f0_iqr"] = float(np.percentile(v, 75)-np.percentile(v, 25))
    f["f0_range_semitones"] = float(12*np.log2(np.percentile(v, 95)/np.percentile(v, 5)))
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=512))
    f["centroid_mean"] = float(librosa.feature.spectral_centroid(S=S, sr=sr)[0].mean())
    flux = np.sqrt(((np.diff(S, axis=1))**2).sum(0))
    f["flux_mean"] = float(flux.mean()); f["flux_cv"] = float(flux.std()/flux.mean())
    # flux expressed relative to the clip's own level: gain-invariant by construction
    f["flux_over_rms"] = f["flux_mean"]/f["rms_mean"]
    ed = librosa.resample(e, orig_sr=sr, target_sr=200)
    ff, P = welch(ed - ed.mean(), fs=200, nperseg=2048)
    band = lambda lo, hi: float(P[(ff >= lo) & (ff < hi)].sum())
    tot = band(0.5, 20) + 1e-12
    f["mod_delta_1_4Hz"] = band(1, 4)/tot; f["mod_theta_4_8Hz"] = band(4, 8)/tot
    return f


res = {}
for tag, normalize in [("raw", False), ("level-matched", True)]:
    res[tag] = {n: features(*load_audio(m, normalize)) for n, m in CLIPS.items()}

keys = list(res["raw"]["Charismatic"].keys())
print("%-24s | %-22s | %-22s" % ("", "SOURCE FILES", "LEVEL-MATCHED (as heard)"))
print("%-24s | %8s %8s %5s | %8s %8s %5s" % ("feature", "Char", "NonChar", "ratio", "Char", "NonChar", "ratio"))
for k in keys:
    row = f"{k:24s} |"
    for tag in ["raw", "level-matched"]:
        c, n = res[tag]["Charismatic"][k], res[tag]["Non-Charismatic"][k]
        row += f" {c:8.4f} {n:8.4f} {c/n if n else float('nan'):5.2f} |"
    print(row)

json.dump(res, open(os.path.join(OUT, "acoustic_features_normalized.json"), "w"), indent=2)
print("\nSaved acoustic_features_normalized.json")
