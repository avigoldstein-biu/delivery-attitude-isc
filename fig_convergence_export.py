"""
Export the three convergence maps for the surface render of Figure 6, panel A.

Each contrast is exported at its reported control stage (see roi_stage_grid.py): Study 1 delivery
after both controls, and both Study 2 manipulations after feature regression. All three are
written unmasked. The figure's purpose is to show that two manipulations land on the same patch
and a third does not, which a common unthresholded colour scale shows directly; masking each map
by its own FDR result would instead show three differently-thresholded sets of blobs, and would
make the cue map blank for a reason a reader could not distinguish from a flat effect.

Also written: a 72-element indicator for the a priori ROI, which the render outlines on the
surface.

Requires: study1_source_myica3_greater_pc0.npz and <DERIV>/s23_stage_maps.npz (roi_stage_grid.py).

Usage: python fig_convergence_export.py
"""
import os, sys
import numpy as np
import scipy.io as sio

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"G:/Barak1"
DERIV = os.path.join(ROOT, "reanalysis")   # reconstruction outputs and intermediate results; point anywhere

ROI = ["Rolandic_Oper_R", "Temporal_Sup_R", "Temporal_Mid_R", "Insula_R"]

s1 = np.load(os.path.join(ROOT, "paper_code", "study1_source_myica3_greater_pc0.npz"))
m23 = np.load(os.path.join(DERIV, "s23_stage_maps.npz"), allow_pickle=True)
labels = [str(x).strip() for x in s1["labels"]]
roi = [labels.index(r) for r in ROI]

PANELS = {
    "conv_s1_delivery": dict(val=np.asarray(s1["beta_both_d"]),
                             title="Study 1: charismatic - non-charismatic, beta",
                             stage="after feature and aperiodic control"),
    "conv_s2_attitude": dict(val=np.asarray(m23["S2_alpha_feat"]),
                             title="Study 2: positive - negative attitude, alpha",
                             stage="after feature control"),
    "conv_s3_cue":      dict(val=np.asarray(m23["S3_alpha_feat"]),
                             title="Study 2: cue - no cue, alpha",
                             stage="after feature control"),
}

roi_ind = np.zeros(72); roi_ind[roi] = 1
out = {k: v["val"].astype(float) for k, v in PANELS.items()}
out["roi"] = roi_ind
out["clim"] = float(max(np.abs(v["val"]).max() for v in PANELS.values()))
sio.savemat(os.path.join(DERIV, "convergence_maps.mat"), out)

print(f"ROI: {ROI}")
for k, v in PANELS.items():
    d = v["val"]
    print(f"{k:18s} whole-brain [{d.min():+.4f}, {d.max():+.4f}]  ROI mean {d[roi].mean():+.4f}  "
          f"({v['stage']})")
print(f"\nsymmetric colour limit {out['clim']:.4f}")
print(f"saved {os.path.join(DERIV, 'convergence_maps.mat')}")
