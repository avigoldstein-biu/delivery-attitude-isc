"""
04 - Trait synchrony. Because the same participants did both tasks (on different videos), we can
ask whether a participant's ISC on one speech predicts their ISC on the other - i.e. whether
'neural synchronizability' is a stable individual trait rather than a property of the stimulus.

Reproduced: alpha ISC on the attitude video correlates with alpha ISC on the cue video (r~.61),
robust to the extreme point (Spearman ~.60; drop-most-extreme ~.56); reliable in theta/alpha/beta
but not delta/gamma (rules out a signal-quality account); alpha survives partialling delta+gamma and
overall ISC.
"""
import os
import numpy as np
from scipy import stats
from lib2 import load, loo, BANDS

ROOT = r"G:/Barak1"
DERIV = os.path.join(ROOT, "reanalysis")   # reconstruction outputs and intermediate results; point anywhere
OUT = os.path.join(DERIV, "trait_synchrony.npz")   # read by fig_trait_synchrony.py

def global_isc(field, inc, gcol):
    X, grp, ids = load(field, inc, gcol)
    z = np.zeros((len(ids), 72, 5))                  # per-subject LOO ISC, computed within group
    for bi in range(5):
        for g in np.unique(grp):
            gi = np.where(grp == g)[0]; z[gi, :, bi] = loo(X[gi, :, :, bi])
    return {int(i): z[k].mean(0) for k, i in enumerate(ids)}   # subj -> global ISC per band (5,)

G2 = global_isc("S2_band", "include_S2", "attitude_grp")
G3 = global_isc("S3_band", "include_S3", "cue_grp")
common = sorted(set(G2) & set(G3))
print(f"n={len(common)} participants completed both tasks\n")

print("cross-video ISC reliability by band (Pearson r, p):")
rel_r = np.zeros(5); rel_p = np.zeros(5)
for bi, b in enumerate(BANDS):
    x = np.array([G2[n][bi] for n in common]); y = np.array([G3[n][bi] for n in common])
    rel_r[bi], rel_p[bi] = stats.pearsonr(x, y)
    print(f"  {b:6s} r={rel_r[bi]:+.2f} p={rel_p[bi]:.4f}{'  *' if rel_p[bi] < .05 else ''}")

ai = BANDS.index("alpha")
x = np.array([G2[n][ai] for n in common]); y = np.array([G3[n][ai] for n in common])
r, p = stats.pearsonr(x, y); sp, _ = stats.spearmanr(x, y)
ix = int(np.argmax(x)); r_nox = stats.pearsonr(np.delete(x, ix), np.delete(y, ix))[0]
print(f"\nalpha trait reliability: Pearson r={r:+.2f} (p={p:.1e}); "
      f"Spearman r={sp:+.2f}; drop most-extreme r={r_nox:+.2f}")

# alpha over and above delta+gamma and overall ISC (partial correlations)
def partial(x, y, covars):
    from numpy.linalg import lstsq
    C = np.column_stack([np.ones_like(x)] + covars)
    rx = x - C @ lstsq(C, x, rcond=None)[0]; ry = y - C @ lstsq(C, y, rcond=None)[0]
    return stats.pearsonr(rx, ry)
d2 = np.array([G2[n][0] for n in common]); g2 = np.array([G2[n][4] for n in common])
d3 = np.array([G3[n][0] for n in common]); g3 = np.array([G3[n][4] for n in common])
o2 = np.array([G2[n].mean() for n in common]); o3 = np.array([G3[n].mean() for n in common])
rp1, pp1 = partial(x, y, [d2, g2, d3, g3])
rp2, pp2 = partial(x, y, [o2, o3])
print(f"  alpha | delta+gamma partialled: r={rp1:+.2f} p={pp1:.3f}")
print(f"  alpha | overall ISC partialled:  r={rp2:+.2f} p={pp2:.3f}")

# saved so that fig_trait_synchrony.py plots computed values rather than transcribed ones
np.savez(OUT, alpha_x=x, alpha_y=y, rel_r=rel_r, rel_p=rel_p, bands=np.array(BANDS),
         r=r, p=p, spearman=sp, r_drop_extreme=r_nox, n=len(common))
print(f"\nsaved {OUT}")
