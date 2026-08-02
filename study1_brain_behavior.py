"""
Brain-behavior: relate per-subject ISC to the questionnaire scores (quest.xlsx), per
condition, per band.
"""
import os, glob, numpy as np, pandas as pd, scipy.io as sio
from scipy import stats
from isc_source import loo, labels
ROOT = r"G:/Barak1"
DERIV = os.path.join(ROOT, "reanalysis")   # reconstruction outputs and intermediate results; point anywhere
SRC = os.path.join(DERIV, "source_myica3")
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
CORE = ["Rolandic_Oper_R", "Temporal_Sup_R", "Temporal_Mid_R", "Cingulum_Mid_L", "Occipital_Sup_L"]
core_idx = [labels.index(r) for r in CORE]

q = pd.read_excel(r"G:/Barak1/quest.xlsx").dropna(subset=["number Eprime"])
q["number Eprime"] = q["number Eprime"].astype(int)
qmap = q.set_index("number Eprime")
BEH = {"charisma": ("MCC", "MCN"), "PANAS_pos": ("PAC", "PAN"), "PANAS_neg": ("NAC", "NAN"),
       "coll_efficacy": ("COEC", "COEN"), "trust": ("TRUC", "TRUN")}

subs = sorted(glob.glob(os.path.join(SRC, "char_*.mat")), key=lambda p: int(os.path.basename(p)[5:-4]))
subj_nums = [int(os.path.basename(s)[5:-4]) for s in subs]        # char_N -> Eprime N
def bvec(col): return np.array([qmap.loc[n, col] if n in qmap.index else np.nan for n in subj_nums], float)

def load(cond):
    A = [sio.loadmat(s)[cond] for s in subs]; T = min(a.shape[1] for a in A)
    return np.stack([a[:, :T].astype(np.float32) for a in A])
C, N = load("Charismatic"), load("Non_Charismatic")

def rp(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    r, p = stats.pearsonr(x[m], y[m]); return f"{r:+.2f}{'*' if p < .05 else ' '}(n{m.sum()})"

def isc_measures(A, bi):                     # per-subject leave-one-out ISC: global, core
    i = loo(A[:, :, :, bi]); return i.mean(1), i[:, core_idx].mean(1)

print("Per-subject ISC vs behaviour (Pearson r; * p<.05), condition-matched, n=%d" % len(q))
for beh, (cc, cn) in BEH.items():
    print(f"\n=== {beh} ===   (Charismatic-cond ISC vs {cc} | Non-charismatic-cond ISC vs {cn})")
    print("%-7s | %-26s | %-26s" % ("band", "GLOBAL ISC (C | NC)", "CORE ISC (C | NC)"))
    bC, bN = bvec(cc), bvec(cn)
    for bi, b in enumerate(BANDS):
        gC, kC = isc_measures(C, bi); gN, kN = isc_measures(N, bi)
        print("%-7s | %s %s | %s %s" % (b, rp(gC, bC), rp(gN, bN), rp(kC, bC), rp(kN, bN)))
