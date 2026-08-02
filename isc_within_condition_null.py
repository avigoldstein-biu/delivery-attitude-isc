"""
Within-condition significance of the inter-subject correlation itself (Section 2.4).

Everything else in this folder tests DIFFERENCES -- between conditions, or between groups. This
tests whether the correlation within a single condition exceeds chance, which is what licenses
treating the baseline ISC values as meaningful at all.

The null circularly shifts the averaged time series that each participant is correlated against,
which destroys the temporal alignment between a participant and the rest of the group while leaving
every signal's own spectrum and autocorrelation untouched. A phase-scrambled or white-noise null
would not: slow, strongly autocorrelated envelopes correlate with one another by construction, and
a null that removes that structure understates chance.

Section 2.4 specifies 1,400 iterations. Rather than recompute a correlation per iteration, the
circular cross-correlation of each participant against the leave-one-out mean is obtained in full by
FFT -- every one of the T possible shifts at once -- and the reported iterations are drawn from it.
The sampled values are therefore exact, not approximations of a shifted correlation.

Shifts within +/- MIN_SHIFT samples of zero are excluded: at 100 Hz after a 0.8 Hz low-pass, a
shift of a few samples leaves the alignment essentially intact and would put genuine signal into
the null.

Reported per band and condition: the observed mean leave-one-out correlation, the 95th percentile
of the null, and the proportion of the 72 regions exceeding it. Values are Pearson r, not Fisher-z,
since that is how Section 2.4 states the threshold.

Requires: <DERIV>/source_myica3 (Study 1) and source_iscex/ (Study 2).

Usage: python isc_within_condition_null.py [study1|study2|both]
"""
import glob, os, re, sys
import numpy as np
import scipy.io as sio

from lib2 import SRC as SRC2, meta

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"G:/Barak1"
DERIV = os.path.join(ROOT, "reanalysis")   # reconstruction outputs and intermediate results; point anywhere

SRC1 = os.path.join(DERIV, "source_myica3")
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
NITER = 1400                      # Section 2.4
MIN_SHIFT = 500                   # 5 s at 100 Hz; smaller shifts do not break the alignment
RNG = np.random.RandomState(0)
WHICH = sys.argv[1] if len(sys.argv) > 1 else "both"


def unit(x):                      # (..., T) -> zero-mean, unit-norm over T
    x = x - x.mean(-1, keepdims=True)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def observed_and_null(A):
    """A (N,72,T) for one band and condition. Returns the observed per-region mean leave-one-out
    correlation, and a null of NITER draws pooled over participants and regions."""
    N, R, T = A.shape
    Au = unit(A.astype(np.float64))
    S = Au.sum(0)
    obs = np.zeros((N, R))
    shifts = RNG.randint(MIN_SHIFT, T - MIN_SHIFT, NITER)
    null = np.zeros((NITER, R))
    Fa = np.fft.rfft(Au, axis=-1)                              # (N,R,T/2+1)
    for i in range(N):
        m = unit((S - Au[i]) / (N - 1))                        # leave-one-out mean, (R,T)
        obs[i] = (Au[i] * m).sum(-1)
        # circular cross-correlation: every shift of m against participant i, at once
        cc = np.fft.irfft(Fa[i] * np.conj(np.fft.rfft(m, axis=-1)), n=T, axis=-1)
        null += cc[:, shifts].T                                # accumulate across participants
    null /= N                                                  # null of the participant-averaged r
    return obs.mean(0), null


def report(A, name):
    """Two readings of Section 2.4's '95th percentile of the null' are reported, because they differ
    by roughly a factor of two and the text does not say which was used:

      per-region  the null of the participant-averaged r in a single region. This is the threshold
                  a per-region ISC value has to clear, and it is the wider of the two.
      global      the null of the r averaged over participants AND all 72 regions, which is the
                  quantity the paper calls global ISC. Averaging over regions narrows the null
                  considerably, so this threshold is the lower one."""
    print(f"\n{name}  (n={A.shape[0]}, T={A.shape[2]}, {NITER:,} iterations)")
    print(f"  {'band':6s} {'observed mean r':>16s} {'null p95':>10s} {'null p95':>10s} "
          f"{'null max':>10s} {'regions >':>11s}")
    print(f"  {'':6s} {'':16s} {'per-region':>10s} {'global':>10s} {'per-region':>10s} "
          f"{'per-reg p95':>11s}")
    for bi, b in enumerate(BANDS):
        obs, null = observed_and_null(A[:, :, :, bi])
        p95 = float(np.percentile(null, 95))
        p95g = float(np.percentile(null.mean(1), 95))
        print(f"  {b:6s} {obs.mean():16.4f} {p95:10.4f} {p95g:10.4f} {null.max():10.4f} "
              f"{int((obs > p95).sum()):6d} / 72", flush=True)


if WHICH in ("study1", "both"):
    subs = sorted([p for p in glob.glob(os.path.join(SRC1, "char_*.mat"))
                   if re.fullmatch(r"char_\d+", os.path.basename(p)[:-4])],
                  key=lambda p: int(os.path.basename(p)[5:-4]))
    print(f"=== Study 1 === {len(subs)} participants")
    for cond in ("Charismatic", "Non_Charismatic", "Silent"):
        A = [sio.loadmat(s)[cond] for s in subs]
        T = min(a.shape[1] for a in A)
        report(np.stack([a[:, :T].astype(np.float32) for a in A]), f"Study 1: {cond}")
        del A

if WHICH in ("study2", "both"):
    subs = sorted(glob.glob(os.path.join(SRC2, "ISCex_*.mat")),
                  key=lambda p: int(os.path.basename(p)[6:-4]))
    for field, inc, gcol, conds in (("S2_band", "include_S2", "attitude_grp", ("Positive", "Negative")),
                                    ("S3_band", "include_S3", "cue_grp", ("With", "Without"))):
        rows = [(int(os.path.basename(s)[6:-4]), s) for s in subs]
        rows = [(n, s, str(meta[meta.subj == n][gcol].iloc[0])) for n, s in rows
                if not meta[meta.subj == n].empty and int(meta[meta.subj == n][inc].iloc[0]) == 1]
        print(f"\n=== Study 2, {field} === {len(rows)} participants")
        for g in conds:                      # leave-one-out is computed within group throughout
            sel = [s for _, s, gg in rows if gg == g]
            A = [sio.loadmat(s)[field] for s in sel]
            T = min(a.shape[1] for a in A)
            report(np.stack([a[:, :T].astype(np.float32) for a in A]), f"Study 2 {field}: {g}")
            del A

print(f"\nSection 2.4 reports the 95th percentile below r = .020 in all bands.")
