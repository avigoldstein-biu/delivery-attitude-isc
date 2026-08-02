"""
Study-1 spectral power across all three conditions, on the
objective-ICA broadband reconstruction.

study1_power_specparam.py reports the charismatic vs non-charismatic contrast but saves
only those two conditions. This runs the same PSD and specparam pipeline for
all three conditions and reports, per band, the region-averaged ANOVA on raw log power and
on the periodic (oscillatory) component, plus the aperiodic exponent.

Usage:  python study1_power_anova.py
"""
import os, sys, glob, numpy as np, scipy.io as sio
from scipy.signal import welch
from scipy import stats
from fooof import FOOOFGroup
ROOT = r"G:/Barak1"
DERIV = os.path.join(ROOT, "reanalysis")   # reconstruction outputs and intermediate results; point anywhere

BB = os.path.join(DERIV, "source_bb_ica")
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 12), "beta": (12, 25), "gamma": (25, 40)}
CONDS = ["Charismatic", "Non_Charismatic", "Silent"]
FSD = 250


def psd_all(cond, subs):
    P = []
    for s in subs:
        R = sio.loadmat(s)[cond]
        f, p = welch(R, fs=FSD, nperseg=4 * FSD, axis=1)
        m = (f >= 1) & (f <= 40); P.append(p[:, m])
    return np.array(P), f[m]


def specparam(P, ff):
    N, R, _ = P.shape
    exp = np.zeros((N, R)); per = np.zeros((N, R, len(BANDS)))
    fg = FOOOFGroup(peak_width_limits=[1, 8], max_n_peaks=6, aperiodic_mode="fixed", verbose=False)
    for i in range(N):
        fg.fit(ff, P[i], freq_range=[1, 40])
        exp[i] = fg.get_params("aperiodic_params", "exponent")
        for r in range(R):
            fr = fg.get_fooof(r)
            ap = 10 ** fr._ap_fit
            periodic = np.clip(10 ** fr.power_spectrum - ap, 0, None)
            for bi, (lo, hi) in enumerate(BANDS.values()):
                per[i, r, bi] = periodic[(fr.freqs >= lo) & (fr.freqs < hi)].mean()
    return exp, per


def rm_anova(X):
    n, k = X.shape; gm = X.mean()
    ss_c = n * ((X.mean(0) - gm) ** 2).sum()
    ss_s = k * ((X.mean(1) - gm) ** 2).sum()
    ss_e = ((X - gm) ** 2).sum() - ss_c - ss_s
    d1, d2 = k - 1, (n - 1) * (k - 1)
    F = (ss_c / d1) / (ss_e / d2)
    return F, stats.f.sf(F, d1, d2), ss_c / (ss_c + ss_e), d2


subs = sorted(glob.glob(os.path.join(BB, "char_*.mat")), key=lambda p: int(os.path.basename(p)[5:-4]))
print(f"{len(subs)} broadband subjects\n")
raw, per, exp = {}, {}, {}
for c in CONDS:
    P, ff = psd_all(c, subs)
    raw[c] = np.stack([P[:, :, (ff >= lo) & (ff < hi)].mean(-1) for lo, hi in BANDS.values()], -1)
    exp[c], per[c] = specparam(P, ff)
    print(f"specparam {c} done")

print("\nRegion-averaged repeated-measures ANOVA across the three conditions:\n")
print(f"  {'band':7s} | {'RAW log power':>28s} | {'PERIODIC power':>28s}")
print(f"  {'':7s} | {'F':>7s} {'p':>9s} {'eta_p2':>7s} | {'F':>7s} {'p':>9s} {'eta_p2':>7s}")
for bi, b in enumerate(BANDS):
    Xr = np.column_stack([np.log(raw[c][:, :, bi].mean(1)) for c in CONDS])
    Xp = np.column_stack([per[c][:, :, bi].mean(1) for c in CONDS])
    Fr, pr, er, d2 = rm_anova(Xr); Fp, pp, ep, _ = rm_anova(Xp)
    print(f"  {b:7s} | {Fr:7.2f} {pr:9.2e} {er:7.3f} | {Fp:7.2f} {pp:9.2e} {ep:7.3f}")
print(f"\n  (df = 2, {d2})")

print("\nCondition means, region-averaged:\n")
print(f"  {'band':7s} " + " ".join(f"{c[:12]:>14s}" for c in CONDS) + "   (raw log power)")
for bi, b in enumerate(BANDS):
    print(f"  {b:7s} " + " ".join(f"{np.log(raw[c][:, :, bi].mean(1)).mean():14.3f}" for c in CONDS))

Xe = np.column_stack([exp[c].mean(1) for c in CONDS])
Fe, pe, ee, _ = rm_anova(Xe)
print(f"\nAperiodic exponent: " + ", ".join(f"{c}={Xe[:, i].mean():.3f}" for i, c in enumerate(CONDS)))
print(f"  ANOVA F(2,{d2}) = {Fe:.2f}, p = {pe:.3f}, eta_p2 = {ee:.3f}")
