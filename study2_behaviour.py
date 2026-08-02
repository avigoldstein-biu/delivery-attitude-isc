"""
05 - Study 2 behaviour: the manipulation check, the ISC-behaviour regression, and the personality
control for the cross-task synchronisation trait.

Covers the numbers Section 4.3 reports outside the ISC analyses themselves:
  - the attitude manipulation check, and the absence of group differences in affect and in memory
    for the lecture
  - alpha correlation predicted from positive affect and attitude, and what happens when group is
    entered
  - whether the cross-task alpha trait of study2_trait_synchrony.py is predicted by the Big Five

SCALE DIRECTION. The ATTITUDE column of study23_labeled.csv is scored so that HIGHER values mean a
LESS favourable evaluation, so the positively-described group scores lower on it. The item wording
is not in this repository, and the direction is inferred from the manipulation rather than from the
instrument; it is printed explicitly below so that a reader can check it against the questionnaire.
Reporting the group means without stating the direction inverts the manipulation check.

The regression is a fixed model, not a stepwise selection. A forward selection from the full
battery retains nothing for alpha: positive affect alone gives R2 = .04, p = .13, so selection
stops at the first step.

Leave-one-out ISC values within a group are not independent of one another, so the OLS F test is
supplemented with a permutation of the behavioural rows, which leaves the ISC dependence structure
untouched.

Requires: source_iscex/, study23_labeled.csv (per-participant questionnaire scores).

Usage: python study2_behaviour.py
"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats as st
from lib2 import load, loo, labels, BANDS

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"G:/Barak1"
QUEST = os.path.join(ROOT, "iscex", "study23_labeled.csv")
NPERM = 10000
RNG = np.random.RandomState(0)

ROI = ["Rolandic_Oper_R", "Temporal_Sup_R", "Temporal_Mid_R", "Insula_R"]   # Section 2.6
roi = [labels.index(r) for r in ROI]
BIG5 = ["extra", "openn", "neurot", "conc", "agree"]

q = pd.read_csv(QUEST).rename(columns={"conc ": "conc"}).set_index("subj")

X, grp, ids = load("S2_band", "include_S2", "attitude_grp")
pos = grp == "Positive"; neg = grp == "Negative"
gP = np.where(pos)[0]; gN = np.where(neg)[0]
print(f"n={len(grp)} (Positive={pos.sum()}, Negative={neg.sum()})")

isc_glob = np.zeros((len(grp), 5)); isc_roi = np.zeros((len(grp), 5))
for bi in range(5):
    z = np.zeros((len(grp), 72))
    for gi in (gP, gN): z[gi] = loo(X[gi, :, :, bi])          # leave-one-out WITHIN group
    isc_glob[:, bi] = z.mean(1); isc_roi[:, bi] = z[:, roi].mean(1)
del X

B = {c: np.array([q.loc[n, c] for n in ids], float)
     for c in ["PANASPOS", "PANASNEG", "ATTITUDE", "memory"] + BIG5}
B["grpP"] = pos.astype(float)

# ---------------------------------------------------------------- manipulation check
print("\n== manipulation check and group comparisons ==")
print("ATTITUDE is reverse-scored: a HIGHER score is a LESS favourable evaluation of the speaker,")
print("so the positively-described group is expected to score LOWER.\n")
print(f"  {'measure':10s} {'Positive':>18s} {'Negative':>18s} {'t':>8s} {'p':>8s} {'d':>7s}")
for c in ["ATTITUDE", "PANASPOS", "PANASNEG", "memory"]:
    a, b = B[c][pos], B[c][neg]
    t, p = st.ttest_ind(a, b)
    sp = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
    print(f"  {c:10s} {a.mean():9.3f} ({a.std(ddof=1):5.3f}) {b.mean():9.3f} ({b.std(ddof=1):5.3f}) "
          f"{t:8.2f} {p:8.4f} {(a.mean()-b.mean())/sp:7.2f}")

# ---------------------------------------------------------------- ISC ~ behaviour
def ols(y, cols):
    """Standardised OLS. Returns R2, the model F test p, and per-predictor beta and p."""
    Xp = np.column_stack([(B[c] - B[c].mean()) / B[c].std() for c in cols])
    yz = (y - y.mean()) / y.std()
    A = np.column_stack([np.ones(len(yz)), Xp])
    bb, *_ = np.linalg.lstsq(A, yz, rcond=None)
    res = yz - A @ bb; n, k = len(yz), A.shape[1]
    se = np.sqrt(np.diag(((res**2).sum() / (n-k)) * np.linalg.pinv(A.T @ A)))
    r2 = 1 - (res**2).sum() / ((yz - yz.mean())**2).sum()
    f = (r2 / (k-1)) / ((1 - r2) / (n - k))
    return r2, st.f.sf(f, k-1, n-k), dict(zip(cols, zip(bb[1:], 2*st.t.sf(np.abs(bb/se)[1:], n-k))))


def perm_p(y, cols, r2_obs):
    """Permute the behavioural rows; the ISC dependence structure is left intact."""
    keep = {c: B[c].copy() for c in cols}
    cnt = 0
    for _ in range(NPERM):
        pm = RNG.permutation(len(y))
        for c in cols: B[c] = keep[c][pm]
        if ols(y, cols)[0] >= r2_obs: cnt += 1
    for c in cols: B[c] = keep[c]
    return (cnt + 1) / (NPERM + 1)


ai = BANDS.index("alpha")
print("\n== alpha correlation predicted from positive affect and attitude ==")
for name, y in [("alpha, global", isc_glob[:, ai]), ("alpha, ROI (4 parcels)", isc_roi[:, ai])]:
    for cols in (["PANASPOS", "ATTITUDE"], ["PANASPOS", "ATTITUDE", "grpP"]):
        r2, p, det = ols(y, cols)
        pp = perm_p(y, cols, r2) if "grpP" not in cols else float("nan")
        line = "  ".join(f"{k} b={v[0]:+.2f} p={v[1]:.3f}" for k, v in det.items())
        tag = f"perm p={pp:.4f}" if pp == pp else "with group entered"
        print(f"  {name:23s} {'+'.join(cols):26s} R2={r2:.3f} p={p:.4f} ({tag})\n"
              f"    {line}")

# ---------------------------------------------------------------- Big Five
print("\n== is alpha correlation predicted by personality? (Section 4.3) ==")
print(f"  {'trait':8s} {'r with alpha global':>21s} {'p':>8s}")
for c in BIG5:
    r, p = st.pearsonr(B[c], isc_glob[:, ai])
    print(f"  {c:8s} {r:21.3f} {p:8.4f}")
