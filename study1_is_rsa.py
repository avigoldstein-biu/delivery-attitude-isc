"""
Inter-subject representational similarity analysis (IS-RSA).
Pairwise between-subject NEURAL similarity (global and core, per band/condition) vs
pairwise BEHAVIOURAL similarity, tested with a Mantel/permutation test. Two behavioural
models (Finn et al. 2020): nearest-neighbour (similar scores -> similar brains) and
Anna Karenina (higher scorers are more alike). Objective-ICA source data; n=39.
"""
import os, glob, numpy as np, pandas as pd, scipy.io as sio
from scipy.stats import rankdata
from isc_source import labels
ROOT = r"G:/Barak1"
DERIV = os.path.join(ROOT, "reanalysis")   # reconstruction outputs and intermediate results; point anywhere
SRC = os.path.join(DERIV, "source_myica3"); BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
# a-priori right auditory-perisylvian ROI -- same set as cross_study_convergence.py
CORE = ["Rolandic_Oper_R", "Temporal_Sup_R", "Temporal_Mid_R", "Insula_R"]
core_idx = [labels.index(r) for r in CORE]
BEH = {"charisma": ("MCC", "MCN"), "PANAS_pos": ("PAC", "PAN"), "coll_efficacy": ("COEC", "COEN"), "trust": ("TRUC", "TRUN")}
NPERM = 10000; rng = np.random.RandomState(0)

q = pd.read_excel(r"G:/Barak1/quest.xlsx").dropna(subset=["number Eprime"]); q["number Eprime"] = q["number Eprime"].astype(int)
qmap = q.set_index("number Eprime")
subs = sorted(glob.glob(os.path.join(SRC, "char_*.mat")), key=lambda p: int(os.path.basename(p)[5:-4]))
nums = [int(os.path.basename(s)[5:-4]) for s in subs]
def load(c): A = [sio.loadmat(s)[c] for s in subs]; T = min(a.shape[1] for a in A); return np.stack([a[:, :T].astype(np.float32) for a in A])
C, N = load("Charismatic"), load("Non_Charismatic")

def neural_sim(A, bi, idx):                        # (Nsub, region, T, band) -> Fisher-z pairwise sim matrix
    sig = A[:, :, :, bi][:, idx].mean(1)           # mean over ROI -> (Nsub, T); subjects stay first
    return np.arctanh(np.clip(np.corrcoef(sig), -.999, .999))

def mantel(neu, mod, keep):                        # Spearman on lower triangle + permutation p
    neu, mod = neu[np.ix_(keep, keep)], mod[np.ix_(keep, keep)]
    n = len(keep); tri = np.tril_indices(n, -1)
    a = rankdata(neu[tri]); b = rankdata(mod[tri]); r0 = np.corrcoef(a, b)[0, 1]
    cnt = 1
    for _ in range(NPERM):
        p = rng.permutation(n); ap = rankdata(neu[np.ix_(p, p)][tri])
        if abs(np.corrcoef(ap, b)[0, 1]) >= abs(r0): cnt += 1
    return r0, cnt/(NPERM+1)

def models(s):                                     # behavioural similarity matrices (z-scored score)
    z = (s-np.nanmean(s))/np.nanstd(s)
    nn = -np.abs(z[:, None]-z[None, :])            # nearest-neighbour
    ak = (z[:, None]+z[None, :])/2                 # Anna Karenina (mean)
    return nn, ak

print("IS-RSA Mantel r (p) — global | core, condition-matched, n=%d, %d perms" % (len(nums), NPERM))
for beh, (cc, cn) in BEH.items():
    print(f"\n=== {beh} ===")
    print("%-7s | %-28s | %-28s" % ("band", "GLOBAL  nn(C) ak(C) | nn(NC) ak(NC)", "CORE  nn(C) ak(C) | nn(NC) ak(NC)"))
    for bi, b in enumerate(BANDS):
        out = {}
        for roiname, idx in [("G", list(range(72))), ("K", core_idx)]:
            cells = []
            for A, col in [(C, cc), (N, cn)]:
                s = np.array([qmap.loc[n, col] if n in qmap.index else np.nan for n in nums], float)
                keep = np.where(np.isfinite(s))[0]; nn, ak = models(s); neu = neural_sim(A, bi, idx)
                rn, pn = mantel(neu, nn, keep); ra, pa = mantel(neu, ak, keep)
                cells.append("%+.2f%s %+.2f%s" % (rn, "*" if pn < .05 else " ", ra, "*" if pa < .05 else " "))
            out[roiname] = " | ".join(cells)
        print("%-7s | %-28s | %-28s" % (b, out["G"], out["K"]))
