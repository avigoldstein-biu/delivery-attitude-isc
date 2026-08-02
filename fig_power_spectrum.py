"""Fig 2a: grand-average power spectrum across regions, per condition, with the
FOOOF/specparam decomposition showing genuine oscillatory peaks above the 1/f aperiodic fit."""
import os
import glob, os, numpy as np, scipy.io as sio
from scipy.signal import welch
from fooof import FOOOF
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = r"G:/Barak1"
DERIV = os.path.join(ROOT, "reanalysis")   # reconstruction outputs and intermediate results; point anywhere
BB = os.path.join(DERIV, "source_bb_ica"); FSD = 250
CONDS = [("Charismatic", "#c0392b"), ("Non_Charismatic", "#2c6fbb"), ("Silent", "#7f8c8d")]
subs = sorted(glob.glob(os.path.join(BB, "char_*.mat")))
print(f"{len(subs)} subjects", flush=True)

gpsd = {}
for cond, _ in CONDS:
    acc = []
    for s in subs:
        R = sio.loadmat(s)[cond].astype(float)                  # (72, T)
        f, p = welch(R, fs=FSD, nperseg=4*FSD, axis=1)
        m = (f >= 1) & (f <= 35)          # trim >35 Hz: source low-pass rolloff (avoids trailing drop)
        acc.append(p[:, m].mean(0))                             # avg across regions
    gpsd[cond] = np.mean(acc, 0); freqs = f[m]                  # avg across subjects
print("PSD computed", flush=True)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
# Panel A: grand-average spectra, all conditions
for cond, col in CONDS:
    ax[0].loglog(freqs, gpsd[cond], color=col, lw=2, label=cond.replace("_", "-"))
ax[0].set_xlabel("Frequency (Hz)"); ax[0].set_ylabel("Power (a.u.)")
ax[0].set_title("Grand-average power spectrum (averaged across 72 regions)", fontsize=11)
ax[0].legend(fontsize=9, frameon=False); ax[0].grid(True, which="both", alpha=0.2)
for band, lo, hi in [("alpha", 8, 12), ("beta", 12, 25)]:
    ax[0].axvspan(lo, hi, color="#f5deb3", alpha=0.25, zorder=0)

# Panel B: FOOOF decomposition of the charismatic grand-average
fm = FOOOF(peak_width_limits=[0.5, 6], min_peak_height=0.05, max_n_peaks=6, verbose=False)
fm.fit(freqs, gpsd["Charismatic"], [1, 35])
ax[1].plot(freqs, np.log10(gpsd["Charismatic"]), "k", lw=2, label="observed spectrum")
ax[1].plot(freqs, fm._ap_fit, "--", color="#c0392b", lw=2, label="aperiodic (1/f) fit")
ax[1].plot(freqs, fm.fooofed_spectrum_, color="#2c6fbb", lw=1.2, alpha=0.8, label="full model fit")
for pk in fm.peak_params_:
    cf = pk[0]
    ax[1].annotate(f"{cf:.1f} Hz", xy=(cf, np.log10(gpsd["Charismatic"])[np.argmin(abs(freqs-cf))]),
                   xytext=(0, 12), textcoords="offset points", ha="center", fontsize=8,
                   arrowprops=dict(arrowstyle="->", color="#555"))
ax[1].set_xscale("log"); ax[1].set_xlabel("Frequency (Hz)"); ax[1].set_ylabel("log10 Power")
ax[1].set_title(f"specparam decomposition (aperiodic exponent = {fm.aperiodic_params_[1]:.2f})\n"
                "an alpha oscillation sits above the 1/f fit", fontsize=11)
ax[1].legend(fontsize=8, frameon=False); ax[1].grid(True, which="both", alpha=0.2)
plt.tight_layout(); fig.savefig(os.path.join(DERIV, "fig_power_spectrum.png"), dpi=160)
print("peaks (CF, PW, BW):", np.round(fm.peak_params_, 2).tolist())
print("saved fig_power_spectrum.png")
