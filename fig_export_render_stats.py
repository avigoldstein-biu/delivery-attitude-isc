"""Export corrected per-region stats to a .mat for the MATLAB/FieldTrip brain render.
72-region order matches the AAL subset [1:20 23:36 43:70 81:90] -> tissue index 1..72."""
import os
import numpy as np, scipy.io as sio

ROOT = r"G:/Barak1"
DERIV = os.path.join(ROOT, "reanalysis")   # reconstruction outputs and intermediate results; point anywhere


def fdr(p, q=0.05):
    p = np.asarray(p); o = np.argsort(p); n = len(p); thr = q*(np.arange(1, n+1))/n
    below = p[o] <= thr; k = np.where(below)[0].max() if below.any() else -1
    s = np.zeros(n, bool); s[o[:k+1]] = k >= 0; return s

# Study-1 maps from the objective-ICA reconstruction, one-sided (study1_isc.py)
s1n = np.load(os.path.join(ROOT, "paper_code", "study1_source_myica3_greater_pc0.npz"))
s1 = {"d_raw": s1n["beta_raw_d"], "sig_raw": s1n["beta_raw_sig"],
      "d_res": s1n["beta_feat_d"], "sig_res": s1n["beta_feat_sig"]}
z = np.load(os.path.join(DERIV, "study2_isc_results.npz"), allow_pickle=True)   # study2_isc_stats.py

panels = {}
# Study 1 beta: raw effect and residual after regression, on the raw-significant regions
panels["s1_beta_raw"] = dict(val=np.asarray(s1["d_raw"]), mask=np.asarray(s1["sig_raw"]).astype(float),
                             clabel="Charismatic - non-charismatic beta ISC (raw)")
# mask by the residual significance, not the raw: painting raw-significant regions with their
# residual values shows near-zero effects as if they survived
panels["s1_beta_res"] = dict(val=np.asarray(s1["d_res"]), mask=np.asarray(s1["sig_res"]).astype(float),
                             clabel="beta ISC after low-level feature regression")
# Study 2 attitude: alpha and theta, FDR one-sided (Positive - Negative)
for b in ["alpha", "theta"]:
    panels[f"s2_{b}"] = dict(val=z[f"S2_{b}_obs"], mask=fdr(z[f"S2_{b}_p1"]).astype(float),
                             clabel=f"Positive - negative {b} ISC")
# Study 1 beta CORE surviving BOTH controls (feature regression + aperiodic removal)
panels["s1_beta_core"] = dict(val=s1n["beta_both_d"], mask=s1n["beta_both_sig"].astype(float),
                              clabel="beta ISC surviving feature + aperiodic controls")
lab = [str(x) for x in s1n["labels"]]
print("surviving-core regions:", [lab[i] for i in np.where(s1n["beta_both_sig"])[0]])
# common colour scale per group
sio.savemat(os.path.join(DERIV, "render_stats.mat"), {k: v for k, v in panels.items()})
for k, v in panels.items():
    print(f"{k}: {int(v['mask'].sum())} regions, val range [{v['val'][v['mask']>0].min():+.3f}, {v['val'][v['mask']>0].max():+.3f}]")
print("saved render_stats.mat")
