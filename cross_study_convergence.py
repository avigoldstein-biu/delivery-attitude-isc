"""Cross-study convergence (Study 1 delivery vs Study 2 attitude vs Study 3 cue).

Tests whether the SAME anatomical network carries the effects across studies, and whether the
frequency channel shifts (S1 delivery = beta; S2 attitude = alpha/theta). All 72 AAL regions are in
identical order across studies. Effect maps (d = condition difference per region):
  S1 = Charismatic - Non_Charismatic 
  S2 = Positive - Negative,  S3 = With - Without cue  
  
Statistics:
  (1) ROI CONVERGENCE: are the S1 and S2 effects concentrated in an a-priori
      ANATOMICAL right auditory-perisylvian ROI (defined independently of both studies)? Mean d in
      ROI vs a null of every same-size region set (exact enumeration). S3 cue is the negative control.
      (1b) reports Study-1-derived core definitions as a sensitivity check only.
  (2) MIN-STATISTIC OVERLAP: regions FDR-sig in BOTH S1(beta) and S2(band); hypergeometric chance.
  (3) LATERALITY of each effect within the same ROI: LI = (R-L)/(R+L) over the four regions.
Negative control throughout: the S3 audience-cue contrast, tested identically.

Usage: python cross_study_convergence.py
"""
import os, sys, numpy as np, scipy.io as sio
from itertools import combinations, islice
from scipy.stats import hypergeom
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"G:/Barak1"
DERIV = os.path.join(ROOT, "reanalysis")   # reconstruction outputs and intermediate results; point anywhere

z = np.load(os.path.join(DERIV, "study2_isc_results.npz"), allow_pickle=True)   # study2_isc_stats.py
labels = [str(x) for x in z["labels"]]

s1 = np.load(os.path.join(ROOT, "paper_code", "study1_source_myica3_greater_pc0.npz"))

def fdr(p, q=0.05):
    p = np.asarray(p); o = np.argsort(p); n = len(p); thr = q*(np.arange(1, n+1))/n
    below = p[o] <= thr; k = np.where(below)[0].max() if below.any() else -1
    sig = np.zeros(n, bool); sig[o[:k+1]] = k >= 0; return sig

# The a-priori core is DERIVED from the Study-1 result rather than hardcoded: the regions whose
# beta ISC condition effect survives BOTH controls (feature regression + aperiodic removal), FDR.
core_idx = list(np.where(s1["beta_both_sig"])[0])
CORE = [labels[i] for i in core_idx]
NCORE = len(CORE)
print(f"a-priori Study-1 core ({NCORE} regions, survive feature + aperiodic control): {CORE}")

s1b_res = s1["beta_feat_d"]      # feature-regressed map, used by the sign check below

# ---------- (2) PRIMARY: enrichment within an a-priori ANATOMICAL ROI ----------
# The reported convergence test uses a region set defined by anatomy, not by either study's
# results: the right auditory-perisylvian ROI, the same set used for the laterality test in (4). Both
# studies are tested in it symmetrically, so neither supplies the other's region definition.
# 

ANAT = ["Rolandic_Oper_R", "Temporal_Sup_R", "Temporal_Mid_R", "Insula_R"]
anat_idx = [labels.index(r) for r in ANAT]
d_both = s1["beta_both_d"]

def enrich(vals, idx, chunk=2_000_000):
    """Mean effect in `idx` against a null of every same-size region set. All C(72,k) sets are
    enumerated rather than sampled, so the p is an exact proportion over the complete reference
    set: 1,028,790 sets for k=4, 13,991,544 for k=5. Returns (observed mean, exceedances, n, p)."""
    obs = vals[idx].mean(); it = combinations(range(72), len(idx)); k = n = 0
    while True:
        buf = list(islice(it, chunk))
        if not buf: break
        m = vals[np.array(buf, np.int8)].mean(1)
        k += int((m >= obs).sum()); n += len(m)
    return obs, k, n, (1 + k) / (n + 1)

print(f"\n(1) PRIMARY CONVERGENCE TEST -- a-priori anatomical ROI: {ANAT}")
print("    (mean effect in ROI vs a null of EVERY same-size region set; exact enumeration)\n")
rows = [("S1 delivery, beta (both controls)", d_both)]
rows += [(f"S2 attitude, {b}", z[f"S2_{b}_obs"]) for b in ["alpha", "theta", "beta"]]
rows += [(f"S3 cue, {b}  (negative control)", z[f"S3_{b}_obs"]) for b in ["alpha", "beta"]]
for lab_, vals in rows:
    o, k, n, p = enrich(np.asarray(vals), anat_idx)
    print(f"     {lab_:38s} mean d={o:+.4f}  {k:7d}/{n}  p={p:.6f}{'  *' if p < .05 else ''}")

# ---------- (1b) SENSITIVITY: Study-1-derived core definitions ----------
# Reported for transparency only. These inherit their definition from the Study-1 result and are
# therefore circular to some degree, and the full surviving set includes occipital and cingulate
# regions that the right-perisylvian prediction does not concern.

print("\n(1b) SENSITIVITY -- Study-1-derived cores (not the reported test)")
peri_surv = [r for r in CORE if r in ("Rolandic_Oper_R", "Temporal_Sup_R", "Temporal_Mid_R")]
for name, regs in [("all surviving regions", CORE), ("perisylvian subset of surviving", peri_surv)]:
    idx = [labels.index(r) for r in regs]
    cells = "  ".join(f"S2-{b}: p={enrich(np.asarray(z[f'S2_{b}_obs']), idx)[3]:.4f}" for b in ["alpha", "theta"])
    ctrl = enrich(np.asarray(z["S3_alpha_obs"]), idx)[3]
    print(f"     {name:32s} (k={len(idx)})  {cells}   S3-alpha(ctrl): p={ctrl:.4f}")

# ---------- (2) minimum-statistic overlap (FDR in both) ----------
print("\n(2) MIN-STATISTIC OVERLAP: regions FDR-sig in BOTH S1(beta, feature + aperiodic) and S2(band)")
s1_sig = s1["beta_both_sig"]                              # S1 beta surviving BOTH controls, FDR
print(f"     S1 beta FDR-sig regions ({s1_sig.sum()}): {[labels[i] for i in np.where(s1_sig)[0]]}")
for b in ["theta", "alpha", "beta"]:
    s2_sig = fdr(z[f"S2_{b}_p1"])                          # one-sided FDR
    both = s1_sig & s2_sig
    K, n1, n2 = 72, s1_sig.sum(), s2_sig.sum()
    exp = n1 * n2 / K
    phyp = hypergeom.sf(both.sum() - 1, K, n1, n2) if n1 and n2 else 1.0
    print(f"     S2-{b:6s} FDR-sig={n2:2d} | overlap={both.sum()} (chance {exp:.2f}, hypergeom p={phyp:.3f}) "
          f"{[labels[i] for i in np.where(both)[0]]}")

# ---------- (3) laterality within the same a-priori ROI ----------
# LI = (sum_R - sum_L) / (sum_R + sum_L) over the four ROI regions; +1 = carried entirely by the
# right hemisphere, 0 = symmetric. Uses the same ROI as (2), so laterality and convergence are
# quantified on one region set rather than two.

print("\n(3) LATERALITY within the a-priori ROI  [LI = (R-L)/(R+L), summed over the four regions]")
BASE = [r.rsplit("_", 1)[0] for r in ANAT]                 # strip the _R suffix -> region stems
for name, d in [("S1 delivery, beta (both controls)", d_both),
                ("S2 attitude, alpha", np.asarray(z["S2_alpha_obs"])),
                ("S2 attitude, theta", np.asarray(z["S2_theta_obs"]))]:
    R = np.array([d[labels.index(b + "_R")] for b in BASE])
    L = np.array([d[labels.index(b + "_L")] for b in BASE])
    li = (R.sum() - L.sum()) / (R.sum() + L.sum())
    print(f"     {name:36s} mean R={R.mean():+.4f}  mean L={L.mean():+.4f}  "
          f"LI={li:+.3f}  R>L in {int((R > L).sum())}/4")

# ---------- sign consistency in the core ----------
print("\n(sign check) S1 vs S2 direction in the core regions (both predicted POSITIVE):")
for c, i in zip(CORE, core_idx):
    print(f"     {c:18s} S1beta d_res={s1b_res[i]:+.4f} | S2 alpha={z['S2_alpha_obs'][i]:+.4f}  theta={z['S2_theta_obs'][i]:+.4f}")
print("\nDone.")
