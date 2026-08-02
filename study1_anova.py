"""
Study-1 global ISC across all three conditions, on the objective-ICA reconstruction.

Repeated-measures ANOVA over the charismatic, non-charismatic and silent conditions with the
post-hoc contrasts. The silent condition carries the same
per-participant ICA solution as the two speech conditions.

Reports, per band:
  - global ISC (72-region mean) for each condition
  - one-way repeated-measures ANOVA across the three conditions
  - the three post-hoc paired contrasts with Cohen's dz
  - the number of regions where each speech condition exceeds silent (paired t, FDR/72)

Usage:  python study1_anova.py [source_myica3|source]
"""
import os, sys, glob, re, gc, numpy as np, scipy.io as sio
from scipy import stats
ROOT = r"G:/Barak1"
DERIV = os.path.join(ROOT, "reanalysis")   # reconstruction outputs and intermediate results; point anywhere

_SET = sys.argv[1] if len(sys.argv) > 1 else "source_myica3"
SRC = os.path.join(DERIV, _SET)
CONDS = ["Charismatic", "Non_Charismatic", "Silent"]
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
OUT = os.path.join(ROOT, "paper_code", f"study1_anova_{_SET}.npz")
labels = [str(x).strip("'") for x in
          sio.loadmat(os.path.join(ROOT, "Charisma", "atlas_info.mat"), squeeze_me=True)["tissuelabel"]]


def loo(arr):
    N = arr.shape[0]; tot = arr.sum(0); out = np.zeros((N, arr.shape[1]))
    for i in range(N):
        o = (tot - arr[i]) / (N - 1)
        a = arr[i] - arr[i].mean(-1, keepdims=True); oo = o - o.mean(-1, keepdims=True)
        r = (a * oo).sum(-1) / (np.sqrt((a ** 2).sum(-1) * (oo ** 2).sum(-1)) + 1e-12)
        out[i] = np.arctanh(np.clip(r, -.999, .999))
    return out


def fdr(p, q=0.05):
    p = np.asarray(p); o = np.argsort(p); n = len(p); thr = q * np.arange(1, n + 1) / n
    below = p[o] <= thr; k = np.where(below)[0].max() if below.any() else -1
    sig = np.zeros(n, bool)
    if k >= 0: sig[o[:k + 1]] = True
    return sig


def rm_anova(X):
    """One-way repeated-measures ANOVA. X (n_subj, n_cond) -> F, p, partial eta squared, df."""
    n, k = X.shape; gm = X.mean()
    ss_c = n * ((X.mean(0) - gm) ** 2).sum()
    ss_s = k * ((X.mean(1) - gm) ** 2).sum()
    ss_e = ((X - gm) ** 2).sum() - ss_c - ss_s
    d1, d2 = k - 1, (n - 1) * (k - 1)
    F = (ss_c / d1) / (ss_e / d2)
    return F, stats.f.sf(F, d1, d2), ss_c / (ss_c + ss_e), d1, d2


def load(cond):
    subs = sorted([p for p in glob.glob(os.path.join(SRC, "char_*.mat"))
                   if re.fullmatch(r"char_\d+", os.path.basename(p)[:-4])],
                  key=lambda p: int(os.path.basename(p)[5:-4]))
    keep = [s for s in subs if cond in sio.whosmat(s) or True]
    A, used = [], []
    for s in keep:
        m = sio.loadmat(s)
        if cond not in m: continue
        A.append(m[cond]); used.append(int(os.path.basename(s)[5:-4]))
    T = min(a.shape[1] for a in A)
    return np.stack([a[:, :T].astype(np.float32) for a in A]), used


# ---- per-condition leave-one-out ISC, one condition in memory at a time ----
iz = {}
for cond in CONDS:
    A, subs = load(cond)
    iz[cond] = {b: loo(A[:, :, :, bi]) for bi, b in enumerate(BANDS)}
    print(f"{cond:16s} n={len(subs)}  T={A.shape[2]}")
    del A; gc.collect()

nsub = min(len(iz[c][BANDS[0]]) for c in CONDS)
print(f"\nGlobal ISC (72-region mean, Fisher-z), n={nsub}\n")
print(f"  {'band':7s} {'Charis':>8s} {'NonChar':>8s} {'Silent':>8s} | "
      f"{'F(2,%d)'%(2*(nsub-1)):>9s} {'p':>9s} {'eta_p2':>7s}")
res = {}
for b in BANDS:
    X = np.column_stack([iz[c][b][:nsub].mean(1) for c in CONDS])
    F, p, eta, d1, d2 = rm_anova(X)
    res[f"{b}_global"] = X
    print(f"  {b:7s} {X[:,0].mean():8.4f} {X[:,1].mean():8.4f} {X[:,2].mean():8.4f} | "
          f"{F:9.3f} {p:9.2e} {eta:7.3f}")

print(f"\nPost-hoc paired contrasts (t, p two-sided, Cohen's dz):\n")
print(f"  {'band':7s} " + " ".join(f"{c:>24s}" for c in
      ["Charis > NonChar", "Charis > Silent", "NonChar > Silent"]))
for b in BANDS:
    X = res[f"{b}_global"]; cells = []
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        t, p = stats.ttest_rel(X[:, i], X[:, j])
        cells.append(f"t={t:+6.2f} p={p:7.1e} dz={t/np.sqrt(len(X)):+5.2f}")
    print(f"  {b:7s} " + " ".join(f"{c:>24s}" for c in cells))

print(f"\nRegions exceeding silent (paired t, one-sided, FDR across 72):\n")
print(f"  {'band':7s} {'Charismatic':>12s} {'Non_Charismatic':>16s}")
for b in BANDS:
    row = []
    for c in ["Charismatic", "Non_Charismatic"]:
        a, s = iz[c][b][:nsub], iz["Silent"][b][:nsub]
        sig = fdr(stats.ttest_rel(a, s, alternative="greater")[1])
        res[f"{b}_{c}_vs_silent_sig"] = sig
        row.append(int(sig.sum()))
    print(f"  {b:7s} {row[0]:12d} {row[1]:16d}")

np.savez(OUT, labels=np.array(labels), **res)
print(f"\nSaved {OUT}")
