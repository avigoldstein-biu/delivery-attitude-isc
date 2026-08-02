"""
Study 1 manipulation check and questionnaire comparisons (Section 3.3).

Five paired comparisons between the charismatic and non-charismatic conditions, one per
questionnaire: perceived charisma (the manipulation check), positive and negative affect, collective
efficacy, and cognition-based trust. Each participant saw both versions, so every comparison is
within-subject.

Columns in quest.xlsx are named <measure><C|N>, C for the charismatic condition and N for the
non-charismatic one:
    MCC / MCN     modified Conger-Kanungo charismatic leadership scale
    PAC / PAN     PANAS, positive affect
    NAC / NAN     PANAS, negative affect
    COEC / COEN   collective efficacy
    TRUC / TRUN   cognition-based trust

EFFECT SIZE. Two conventions are reported, because a paired design admits both and they differ
substantially here. dz divides the mean difference by the standard deviation OF THE DIFFERENCE and
is the one that corresponds to the paired t (dz = t / sqrt(n)). d_av divides it by the average of
the two conditions' standard deviations, ignoring the pairing, and is the one comparable to a
between-groups d. Which the manuscript intends should be stated, since for a strongly correlated
pair the two can differ by a factor of two or more.

The pilot rating study of Section 3.2.2 (30 raters, Cronbach's alpha = .784) used a separate online
sample whose item-level responses are not in this repository, so that value cannot be recomputed
here.

Requires: quest.xlsx.

Usage: python study1_behaviour.py
"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats as st

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"G:/Barak1"
QUEST = os.path.join(ROOT, "quest.xlsx")

MEASURES = [("perceived charisma", "MCC", "MCN"),
            ("positive affect (PANAS)", "PAC", "PAN"),
            ("negative affect (PANAS)", "NAC", "NAN"),
            ("collective efficacy", "COEC", "COEN"),
            ("cognition-based trust", "TRUC", "TRUN")]

q = pd.read_excel(QUEST).dropna(subset=["number Eprime"])
print(f"n = {len(q)} participants\n")

print(f"{'measure':26s} {'charismatic':>16s} {'non-charismatic':>18s} {'t':>7s} {'p':>9s} "
      f"{'dz':>7s} {'d_av':>7s} {'r':>6s}")
for name, cc, cn in MEASURES:
    a = q[cc].astype(float).values; b = q[cn].astype(float).values
    ok = np.isfinite(a) & np.isfinite(b); a, b = a[ok], b[ok]
    t, p = st.ttest_rel(a, b)
    diff = a - b
    dz = diff.mean() / diff.std(ddof=1)                       # matches the paired t
    d_av = diff.mean() / ((a.std(ddof=1) + b.std(ddof=1)) / 2)   # ignores the pairing
    r = st.pearsonr(a, b)[0]
    print(f"{name:26s} {a.mean():7.2f} ({a.std(ddof=1):4.2f}) {b.mean():9.2f} ({b.std(ddof=1):4.2f}) "
          f"{t:7.2f} {p:9.4f} {dz:7.3f} {d_av:7.3f} {r:6.2f}")

print(f"\ndf = {int(ok.sum()) - 1} for every comparison; p is two-tailed.")
print("dz = mean difference / SD of the difference (= t / sqrt(n)); d_av uses the average of the\n"
      "two conditions' SDs. r is the correlation between conditions: the more strongly the two are\n"
      "correlated, the larger dz is relative to d_av.")
