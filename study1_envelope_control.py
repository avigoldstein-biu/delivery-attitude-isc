"""
Envelope-MEG coupling and sensor-level band power (Supplementary S3), from the extraction
written by study1_sensor_extract.py. Sensor level, all 40 participants.

  1. Envelope coupling, against a circular-shift null:
       env_track_r    fast (1-8 Hz) envelope tracking over auditory sensors
       slowpow_env_r  the 0.8 Hz low-passed theta power envelope -- the signal that enters
                      the ISC analysis -- against the acoustic envelope
  2. Band power, charismatic vs non-charismatic, all five bands including delta.
  3. specparam: periodic vs aperiodic, and whether condition differences are broadband.
"""
import os, numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from fooof import FOOOFGroup
ROOT = r"G:/Barak1"
DERIV = os.path.join(ROOT, "reanalysis")   # reconstruction outputs and intermediate results; point anywhere

OUT = DERIV
d = np.load(os.path.join(OUT, "meg_extract.npz"), allow_pickle=True)
ff = d["psd_freqs"]; bands = list(d["bands"])
csub = list(d["Charismatic_subj"]); nsub = list(d["NonCharismatic_subj"])
common = [s for s in csub if s in nsub]
ci = [csub.index(s) for s in common]; ni = [nsub.index(s) for s in common]
print(f"n paired subjects = {len(common)}")

def paired(cvals, nvals, label):
    t, p = stats.ttest_rel(cvals, nvals)
    print(f"  {label:16s} C={cvals.mean():.4f}  NC={nvals.mean():.4f}  "
          f"t({len(cvals)-1})={t:+.2f} p={p:.4f}  dz={t/np.sqrt(len(cvals)):+.2f}")
    return t, p

print("\n=== B: envelope coupling (correct alignment; tested vs circular-shift NULL) ===")
for key, lab in [("env_track_r", "env tracking r"), ("slowpow_env_r", "2.3.2 slowpow r")]:
    obs = np.concatenate([d[f"Charismatic_{key}"][ci], d[f"Non-Charismatic_{key}"][ni]])
    null = np.concatenate([d[f"Charismatic_{key}_null"][ci], d[f"Non-Charismatic_{key}_null"][ni]])
    t1, p1 = stats.ttest_rel(obs, null)
    print(f"[{lab}] obs={obs.mean():.4f}  null={null.mean():.4f}  obs>null: t={t1:.2f} p={p1:.1e}  dz={t1/np.sqrt(len(obs)):+.2f}")
    paired(d[f"Charismatic_{key}"][ci], d[f"Non-Charismatic_{key}"][ni], "C vs NC")

print("\n=== C: band power, Charismatic vs Non-Charismatic (incl. DELTA) ===")
cbp = d["Charismatic_bandpow"][ci]; nbp = d["Non-Charismatic_bandpow"][ni]
for bi, b in enumerate(bands):
    paired(np.log(cbp[:, bi]), np.log(nbp[:, bi]), f"log {b}")

print("\n=== D: FOOOF periodic/aperiodic ===")
def fit(psd):
    fg = FOOOFGroup(peak_width_limits=[1, 8], max_n_peaks=6, aperiodic_mode="fixed", verbose=False)
    fg.fit(ff, psd, freq_range=[1, 40])
    exps = fg.get_params("aperiodic_params", "exponent")
    offs = fg.get_params("aperiodic_params", "offset")
    peaks = fg.get_params("peak_params", "CF")  # center freqs of all peaks
    return exps, offs, peaks
cexp, coff, cpk = fit(d["Charismatic_psd"][ci])
nexp, noff, npk = fit(d["Non-Charismatic_psd"][ni])
paired(cexp, nexp, "aperiodic exp")
paired(coff, noff, "aperiodic off")
for b, (lo, hi) in {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 12), "beta": (12, 25), "gamma": (25, 40)}.items():
    fr = np.mean((cpk >= lo) & (cpk < hi))  # fraction of detected peaks in band (Charismatic)
    print(f"  oscillatory peaks in {b:6s}: {fr*100:4.1f}% of detected peaks (Charismatic)")

# figure
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
ax[0].bar([f"{b}\n(track)" for b in ["env", "2.3.2"]],
          [np.concatenate([d[f'Charismatic_env_track_r'][ci], d['Non-Charismatic_env_track_r'][ni]]).mean(),
           np.concatenate([d['Charismatic_slowpow_env_r'][ci], d['Non-Charismatic_slowpow_env_r'][ni]]).mean()])
ax[0].set_title("B: envelope coupling (pooled)"); ax[0].set_ylabel("|r|")
x = np.arange(len(bands))
ax[1].bar(x-0.2, np.log(cbp).mean(0), 0.4, label="Charismatic")
ax[1].bar(x+0.2, np.log(nbp).mean(0), 0.4, label="Non-Charismatic")
ax[1].set_xticks(x); ax[1].set_xticklabels(bands); ax[1].legend(); ax[1].set_title("C: log band power");
ax[2].loglog(ff, d["Charismatic_psd"][ci].mean(0), label="Charismatic")
ax[2].loglog(ff, d["Non-Charismatic_psd"][ni].mean(0), label="Non-Charismatic")
ax[2].set_title(f"D: mean PSD (aperiodic exp C={cexp.mean():.2f} NC={nexp.mean():.2f})")
ax[2].set_xlabel("Hz"); ax[2].legend()
plt.tight_layout(); fig.savefig(os.path.join(OUT, "analysis_BCD.png"), dpi=110)
print("\nSaved analysis_BCD.png")
