"""
Source-level power with specparam.
From broadband region time series (source_bb_ica/*.mat): Welch PSD per region/condition,
fit specparam (1-40 Hz), and split each region's power into APERIODIC (1/f) and
PERIODIC (oscillatory, above 1/f). Then Charismatic vs Non-Charismatic:
 - raw log band power (does the alpha/beta posterior reduction replicate?)
 - aperiodic exponent/offset (is the difference a broadband 1/f shift?)
 - PERIODIC band power (is the alpha/beta reduction in the oscillatory component?)
FDR across 72 regions; posterior ROI highlighted.
"""
import os, glob, numpy as np, scipy.io as sio
from scipy.signal import welch
from scipy import stats
from fooof import FOOOFGroup
from isc_source import labels, GROUPS, fdr, SRC

BB = os.path.join(os.path.dirname(SRC), "source_bb_ica")   # objective-ICA broadband (ft_source_broadband_ica.m)
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 12), "beta": (12, 25), "gamma": (25, 40)}
CONDS = ["Charismatic", "Non_Charismatic", "Silent"]
FSD = 250

def psd_all(cond, subs):
    P = []
    for s in subs:
        R = sio.loadmat(s)[cond]                       # (72, T)
        f, p = welch(R, fs=FSD, nperseg=4*FSD, axis=1)
        m = (f >= 1) & (f <= 40); P.append(p[:, m])
    return np.array(P), f[m]                            # (N,72,nf), freqs

def specparam(P, ff):
    N, R, _ = P.shape
    exp = np.zeros((N, R)); off = np.zeros((N, R)); per = np.zeros((N, R, len(BANDS)))
    fg = FOOOFGroup(peak_width_limits=[1, 8], max_n_peaks=6, aperiodic_mode="fixed", verbose=False)
    for i in range(N):
        fg.fit(ff, P[i], freq_range=[1, 40])
        off[i] = fg.get_params("aperiodic_params", "offset")
        exp[i] = fg.get_params("aperiodic_params", "exponent")
        for r in range(R):
            fr = fg.get_fooof(r)
            ap = 10**fr._ap_fit                        # aperiodic (log10 power -> linear), on fr.freqs
            periodic = np.clip(10**fr.power_spectrum - ap, 0, None)
            for bi, (b, (lo, hi)) in enumerate(BANDS.items()):
                per[i, r, bi] = periodic[(fr.freqs >= lo) & (fr.freqs < hi)].mean()
    return exp, off, per

def main():
    subs = sorted(glob.glob(os.path.join(BB, "char_*.mat")), key=lambda p: int(os.path.basename(p)[5:-4]))
    print(f"{len(subs)} broadband subjects")
    data = {}
    for c in CONDS:
        P, ff = psd_all(c, subs); data[c] = specparam(P, ff)
        print(f"specparam {c} done")
    exp = {c: data[c][0] for c in CONDS}; off = {c: data[c][1] for c in CONDS}; per = {c: data[c][2] for c in CONDS}
    # raw band power (mean PSD in band) for comparison
    rawpow = {}
    for c in CONDS:
        P, ff = psd_all(c, subs)
        rawpow[c] = np.stack([P[:, :, (ff >= lo) & (ff < hi)].mean(-1) for lo, hi in BANDS.values()], -1)
    post = [i for i, l in enumerate(labels) if any(k in l for k in GROUPS["occipital"]+GROUPS["parietal"])]
    bn = list(BANDS)
    print("\n=== Charismatic vs Non-Charismatic (posterior ROI, paired t; n regions FDR-sig) ===")
    print("%-6s | raw log-power       | PERIODIC power        | aperiodic exp" % "band")
    for bi, b in enumerate(bn):
        tr, pr = stats.ttest_rel(np.log(rawpow["Charismatic"][:, :, bi]), np.log(rawpow["Non_Charismatic"][:, :, bi]))
        tp, pp = stats.ttest_rel(np.log(per["Charismatic"][:, :, bi]+1e-9), np.log(per["Non_Charismatic"][:, :, bi]+1e-9))
        dr = np.log(rawpow["Charismatic"][:, post, bi]).mean()-np.log(rawpow["Non_Charismatic"][:, post, bi]).mean()
        dp = (np.log(per["Charismatic"][:, post, bi]+1e-9)-np.log(per["Non_Charismatic"][:, post, bi]+1e-9)).mean()
        print("%-6s | d=%+.3f  FDRsig=%2d | d=%+.3f  FDRsig=%2d | (see below)"
              % (b, dr, fdr(pr).sum(), dp, fdr(pp).sum()))
    te, pe = stats.ttest_rel(exp["Charismatic"], exp["Non_Charismatic"])
    print("\naperiodic exponent C vs NC: mean %.3f vs %.3f, posterior d=%+.3f, FDR-sig regions=%d"
          % (exp["Charismatic"].mean(), exp["Non_Charismatic"].mean(),
             exp["Charismatic"][:, post].mean()-exp["Non_Charismatic"][:, post].mean(), fdr(pe).sum()))
    np.savez(os.path.join(os.path.dirname(SRC), "source_power_specparam.npz"),
             exp_C=exp["Charismatic"], exp_N=exp["Non_Charismatic"],
             per_C=per["Charismatic"], per_N=per["Non_Charismatic"],
             rawC=rawpow["Charismatic"], rawN=rawpow["Non_Charismatic"], bands=bn)
    print("saved source_power_specparam.npz")

if __name__ == "__main__":
    main()
