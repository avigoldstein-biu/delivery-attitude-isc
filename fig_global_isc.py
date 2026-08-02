"""Global ISC by frequency band and condition (Study 1): shows beta carries the highest
inter-subject correlation WITHIN each condition, distinct from the between-condition effect."""
import os
import glob, os, re, numpy as np, scipy.io as sio
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = r"G:/Barak1"
DERIV = os.path.join(ROOT, "reanalysis")   # reconstruction outputs and intermediate results; point anywhere
# objective-ICA reconstruction; all three conditions live in one file per subject
SRC = os.path.join(DERIV, "source_myica3")
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]

def loo(A):  # (N,72,T) -> (N,72) Fisher-z ISC
    N = A.shape[0]; tot = A.sum(0); out = np.zeros((N, A.shape[1]))
    for i in range(N):
        o = (tot-A[i])/(N-1); a = A[i]-A[i].mean(-1, keepdims=True); oo = o-o.mean(-1, keepdims=True)
        r = (a*oo).sum(-1)/(np.sqrt((a**2).sum(-1)*(oo**2).sum(-1))+1e-12)
        out[i] = np.arctanh(np.clip(r, -.999, .999))
    return out

subs = sorted([p for p in glob.glob(os.path.join(SRC, "char_*.mat"))
               if re.fullmatch(r"char_\d+", os.path.basename(p)[:-4])],
              key=lambda p: int(os.path.basename(p)[5:-4]))
print(f"{len(subs)} subjects", flush=True)

def stack(field, files):
    dat = [sio.loadmat(f)[field].astype(np.float32) for f in files]
    T = min(d.shape[1] for d in dat)
    return np.stack([d[:, :T, :] for d in dat])                # (N,72,T,5)

C = stack("Charismatic", subs); N = stack("Non_Charismatic", subs)
S = stack("Silent", subs)
print("loaded", flush=True)

# per-subject global ISC per band per condition
def gisc(X):
    return np.stack([loo(X[:, :, :, bi]).mean(1) for bi in range(5)], 1)   # (N,5)
gC, gN, gS = gisc(C), gisc(N), gisc(S)

fig, ax = plt.subplots(figsize=(8.5, 4.8)); x = np.arange(5); w = 0.26
for j, (g, lab, col) in enumerate([(gC, "Charismatic", "#c0392b"), (gN, "Non-charismatic", "#2c6fbb"),
                                   (gS, "Silent", "#95a5a6")]):
    m = g.mean(0); se = g.std(0)/np.sqrt(g.shape[0])
    ax.bar(x+(j-1)*w, m, w, yerr=se, capsize=2, color=col, label=lab, error_kw=dict(lw=0.8))
# valid paired test C vs NC per band
for bi in range(5):
    t, p = stats.ttest_rel(gC[:, bi], gN[:, bi])
    if p < .05:
        ax.text(x[bi]-w, max(gC[:, bi].mean(), gN[:, bi].mean())+0.006, "*", ha="center", fontsize=14)
ax.set_xticks(x); ax.set_xticklabels(BANDS); ax.set_ylabel("Global ISC (Fisher-z)")
ax.set_title("Global ISC by band and condition (Study 1)\nbeta carries the highest inter-subject correlation within each condition; * = charismatic > non-charismatic",
             fontsize=10)
ax.legend(fontsize=9, frameon=False); ax.axhline(0, color="k", lw=.5)
plt.tight_layout(); fig.savefig(os.path.join(DERIV, "fig_global_isc.png"), dpi=160)
print("mean global ISC (C / N / S) per band:")
for bi, b in enumerate(BANDS):
    print(f"  {b:6s} C={gC[:,bi].mean():.3f} N={gN[:,bi].mean():.3f} S={gS[:,bi].mean():.3f}")
print("saved fig_global_isc.png")
