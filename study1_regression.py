"""
Study-1 brain-behaviour regression: global ISC predicted from the
questionnaire battery, on the objective-ICA reconstruction.

Predictors are condition-matched: for the charismatic condition the charismatic-condition
ratings, for the non-charismatic condition the non-charismatic ratings.

Usage:  python study1_regression.py [source_myica3|source]
"""
import os, sys, glob, re, numpy as np, pandas as pd, scipy.io as sio
from scipy import stats
ROOT = r"G:/Barak1"
DERIV = os.path.join(ROOT, "reanalysis")   # reconstruction outputs and intermediate results; point anywhere

_SET = sys.argv[1] if len(sys.argv) > 1 else "source_myica3"
SRC = os.path.join(DERIV, _SET)
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
# questionnaire columns, (charismatic-condition, non-charismatic-condition)
PRED = {"charisma": ("MCC", "MCN"), "PANAS_pos": ("PAC", "PAN"), "PANAS_neg": ("NAC", "NAN"),
        "coll_efficacy": ("COEC", "COEN"), "trust": ("TRUC", "TRUN")}
CONDS = ["Charismatic", "Non_Charismatic"]


def loo(arr):
    N = arr.shape[0]; tot = arr.sum(0); out = np.zeros((N, arr.shape[1]))
    for i in range(N):
        o = (tot - arr[i]) / (N - 1)
        a = arr[i] - arr[i].mean(-1, keepdims=True); oo = o - o.mean(-1, keepdims=True)
        r = (a * oo).sum(-1) / (np.sqrt((a ** 2).sum(-1) * (oo ** 2).sum(-1)) + 1e-12)
        out[i] = np.arctanh(np.clip(r, -.999, .999))
    return out


def ols(y, X, names):
    """Least squares with standardised predictors -> betas, t, p, R2, adjusted R2."""
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    y, X = y[ok], X[ok]
    Xz = (X - X.mean(0)) / X.std(0); yz = (y - y.mean()) / y.std()
    A = np.column_stack([np.ones(len(yz)), Xz])
    beta, *_ = np.linalg.lstsq(A, yz, rcond=None)
    resid = yz - A @ beta
    n, k = len(yz), A.shape[1]
    s2 = (resid ** 2).sum() / (n - k)
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(A.T @ A)))
    t = beta / se; p = 2 * stats.t.sf(np.abs(t), n - k)
    r2 = 1 - (resid ** 2).sum() / ((yz - yz.mean()) ** 2).sum()
    return beta[1:], t[1:], p[1:], r2, 1 - (1 - r2) * (n - 1) / (n - k), int(n)


subs = sorted([p for p in glob.glob(os.path.join(SRC, "char_*.mat"))
               if re.fullmatch(r"char_\d+", os.path.basename(p)[:-4])],
              key=lambda p: int(os.path.basename(p)[5:-4]))
nums = [int(os.path.basename(s)[5:-4]) for s in subs]
q = pd.read_excel(os.path.join(ROOT, "quest.xlsx")).dropna(subset=["number Eprime"])
q["number Eprime"] = q["number Eprime"].astype(int); qmap = q.set_index("number Eprime")

iz = {}
for cond in CONDS:
    A = [sio.loadmat(s)[cond] for s in subs]; T = min(a.shape[1] for a in A)
    A = np.stack([a[:, :T].astype(np.float32) for a in A])
    iz[cond] = {b: loo(A[:, :, :, bi]).mean(1) for bi, b in enumerate(BANDS)}
    del A

names = list(PRED)
for cond, ci in [("Charismatic", 0), ("Non_Charismatic", 1)]:
    X = np.column_stack([[qmap.loc[n, PRED[k][ci]] if n in qmap.index else np.nan for n in nums]
                         for k in names]).astype(float)
    print(f"\n=== {cond} : global ISC ~ {' + '.join(names)} ===")
    print(f"  {'band':8s} {'R2':>6s} {'adjR2':>7s} {'n':>4s} | " + " ".join(f"{k:>14s}" for k in names))
    rows = {b: iz[cond][b] for b in BANDS}
    rows["MEAN"] = np.mean([iz[cond][b] for b in BANDS], axis=0)
    for b, y in rows.items():
        beta, t, p, r2, ar2, n = ols(np.asarray(y, float), X, names)
        cells = " ".join(f"{bb:+7.3f}{'*' if pp < .05 else ' '}({pp:4.2f})" for bb, pp in zip(beta, p))
        print(f"  {b:8s} {r2:6.3f} {ar2:7.3f} {n:4d} | {cells}")
print("\nbeta = standardised coefficient; p in parentheses; * p < .05")
