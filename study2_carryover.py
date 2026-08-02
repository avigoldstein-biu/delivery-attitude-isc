"""
Was the crossed design clean? (Section 4.3)

Study 2 ran two manipulations in sequence within one session: the attitude induction with the first
lecture, the audience cue with the second. Because they were crossed and counterbalanced, each
factor should be inert on the other's task. Two checks, both null under the manuscript's account:

  attitude group tested on the AUDIENCE-CUE task   -- the carry-over the manuscript reports
  cue group tested on the ATTITUDE task            -- the complement, which also has to hold for
                                                      the crossed design to be clean, and which is
                                                      the stronger test of the two because the
                                                      attitude task came first

Both are tested exactly as the main contrasts are: group labels permuted with leave-one-out ISC
re-computed within the permuted groups. Neither direction was predicted, so the two-sided p is the
one to read; the one-sided p is printed only for comparability with the main tables.

Requires: source_iscex/ from matlab/source_reconstruction_iscex.m.

Usage: python study2_carryover.py
"""
import gc, sys
import numpy as np
from lib2 import load, perm_between, perm_global, fdr, labels, BANDS, NPERM

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RNG = np.random.RandomState(5)

CHECKS = [
    ("S3_band", "include_S3", "attitude_grp", "Positive", "Negative",
     "attitude group on the AUDIENCE-CUE task (the carry-over check)"),
    ("S2_band", "include_S2", "cue_grp", "With", "Without",
     "cue group on the ATTITUDE task (the complement)"),
]

for field, inc, gcol, A, B, title in CHECKS:
    X = None; gc.collect()
    X, grp, _ = load(field, inc, gcol)
    gA = np.where(grp == A)[0]; gB = np.where(grp == B)[0]
    print(f"\n===== {title} =====")
    print(f"  {field}, n={X.shape[0]} ({A}={len(gA)}, {B}={len(gB)}), {NPERM:,} permutations")
    print(f"  {'band':6s} {'global d':>9s} {'global p':>9s} {'FDR 2s':>7s} {'FDR 1s':>7s} "
          f"{'max |d|':>8s} {'at':>22s}")
    for bi, b in enumerate(BANDS):
        d, p, _ = perm_global(X[:, :, :, bi], gA, gB, one_sided=False, rng=RNG)
        obs, p2, p1 = perm_between(X[:, :, :, bi], gA, gB, rng=RNG)
        j = int(np.argmax(np.abs(obs)))
        print(f"  {b:6s} {d:+9.4f} {p:9.4f} {int(fdr(p2).sum()):7d} {int(fdr(p1).sum()):7d} "
              f"{obs[j]:+8.4f} {labels[j]:>22s}", flush=True)
    del X

print("\nA clean crossed design requires both checks to be null. The global p here is two-sided,\n"
      "since neither direction was predicted.")
