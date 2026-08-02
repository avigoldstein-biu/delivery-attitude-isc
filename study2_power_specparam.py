"""
04 - Spectral power across groups for Studies 2 and 3, with the periodic/aperiodic split.

The ISC scripts test whether listeners' responses became more similar. This one tests whether
overall power differed, which is the control for a trivial explanation: a group difference in
signal amplitude or in the 1/f slope could change ISC without any change in shared processing.

From the broadband region series (S2_bb, S3_bb at 250 Hz): Welch PSD per region, specparam fit
over 1-40 Hz, then each region's band power split into aperiodic (1/f) and periodic
(oscillatory) parts. Groups are compared on raw log band power, on the aperiodic exponent and
offset, and on periodic band power, with a group-label permutation null and FDR across the 72
regions within each band.

This is the source of the statement that spectral power did not differ between the attitude
groups in any band, and likewise for the audience cue.

Requires: source_iscex/ from matlab/source_reconstruction_iscex.m (broadband fields).
"""
import glob, os, sys
import numpy as np
import scipy.io as sio
from scipy.signal import welch
from fooof import FOOOFGroup

from lib2 import SRC, meta, labels, fdr, BANDS

FSD = 250                                                  # broadband sampling rate
EDGES = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 12), "beta": (12, 25), "gamma": (25, 40)}
NPERM = 10000
RNG = np.random.RandomState(0)
# the Study-1 delivery core, for the mean periodic difference in those regions
CORE1 = ["Rolandic_Oper_R", "Temporal_Sup_R", "Temporal_Mid_R", "Cingulum_Mid_L", "Occipital_Sup_L"]
core_idx = [labels.index(r) for r in CORE1]


def included(inc, gcol):
    """The subjects included in one experiment: (id, path, group). Same selection as lib2.load,
    but the file is opened one at a time and not truncated to a common length -- each PSD is
    computed from that participant's full recording."""
    subs = sorted(glob.glob(os.path.join(SRC, "ISCex_*.mat")),
                  key=lambda p: int(os.path.basename(p)[6:-4]))
    rows = [(int(os.path.basename(s)[6:-4]), s) for s in subs]
    return [(n, s, str(meta[meta.subj == n][gcol].iloc[0])) for n, s in rows
            if not meta[meta.subj == n].empty and int(meta[meta.subj == n][inc].iloc[0]) == 1]


def analyze(field, inc, gcol, A, B, title):
    keep = included(inc, gcol); grp = np.array([g for _, _, g in keep]); n = len(keep)
    exp = np.zeros((n, 72)); off = np.zeros((n, 72))
    raw = np.zeros((n, 72, len(EDGES))); per = np.zeros((n, 72, len(EDGES)))
    fg = FOOOFGroup(peak_width_limits=[1, 8], max_n_peaks=6, aperiodic_mode="fixed", verbose=False)

    for i, (sid, s, g) in enumerate(keep):
        R = sio.loadmat(s)[field]                                          # (72, T)
        f, P = welch(R, fs=FSD, nperseg=4*FSD, axis=1)
        m = (f >= 1) & (f <= 40); P = P[:, m]; ff = f[m]
        for bi, (lo, hi) in enumerate(EDGES.values()):
            raw[i, :, bi] = P[:, (ff >= lo) & (ff < hi)].mean(1)
        fg.fit(ff, P, freq_range=[1, 40])
        off[i] = fg.get_params("aperiodic_params", "offset")
        exp[i] = fg.get_params("aperiodic_params", "exponent")
        for r in range(72):                                                # periodic = spectrum - 1/f fit
            fr = fg.get_fooof(r)
            resid = np.clip(10**fr.power_spectrum - 10**fr._ap_fit, 0, None)
            for bi, (lo, hi) in enumerate(EDGES.values()):
                per[i, r, bi] = resid[(fr.freqs >= lo) & (fr.freqs < hi)].mean()
        print(f"  [{i+1:2d}/{n}] ISCex_{sid} ({g})", flush=True)

    gA = grp == A; gB = grp == B

    def grptest(M):                                    # (N,72) -> observed difference, two-sided perm p
        obs = M[gA].mean(0) - M[gB].mean(0); N = len(M); nA = gA.sum()
        null = np.array([M[(pi := RNG.permutation(N))[:nA]].mean(0) - M[pi[nA:]].mean(0)
                         for _ in range(NPERM)])
        return obs, (np.abs(null) >= np.abs(obs)).mean(0)

    print(f"\n===== {title} =====  n={n} ({A}={gA.sum()}, {B}={gB.sum()})")
    oe, pe = grptest(exp)
    print(f"  aperiodic EXPONENT: {fdr(pe).sum()} sig regions (delta range {oe.min():+.2f}..{oe.max():+.2f})")
    oo, po = grptest(off)
    print(f"  aperiodic OFFSET:   {fdr(po).sum()} sig regions")
    def named(obs, p):                                 # FDR-surviving regions, strongest first
        sig = fdr(p)
        return [f"{labels[i]} ({obs[i]:+.3g}, p={p[i]:.4f})" for i in np.argsort(-np.abs(obs)) if sig[i]]

    for bi, b in enumerate(EDGES):
        orw, prw = grptest(np.log10(raw[:, :, bi] + 1e-30))
        opr, ppr = grptest(per[:, :, bi])
        print(f"  {b:6s}: raw-logpow {fdr(prw).sum():2d} sig (min p={prw.min():.3f}) | "
              f"PERIODIC-pow {fdr(ppr).sum():2d} sig (min p={ppr.min():.3f}) | "
              f"core periodic delta={opr[core_idx].mean():+.3g}")
        for tag, o, pv in (("raw", orw, prw), ("periodic", opr, ppr)):
            for r in named(o, pv):
                print(f"           {tag:8s} {r}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    analyze("S2_bb", "include_S2", "attitude_grp", "Positive", "Negative",
            "STUDY 2 power / specparam (Positive - Negative)")
    analyze("S3_bb", "include_S3", "cue_grp", "With", "Without",
            "STUDY 3 power / specparam (With - Without)")
