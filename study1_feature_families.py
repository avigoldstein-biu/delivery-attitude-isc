"""
08 - Which feature family carries the beta ISC reduction? The main control (study1_isc.py) regresses the
whole 10-feature battery at once, which leaves it ambiguous whether the reduction is driven by
the amplitude envelope or by the prosodic/spectral features. This splits the battery into three
families and regresses each separately, each against its own phase-scrambled null (matched
column count and spectrum), so families with different degrees of freedom are comparable.

  envelope family : envelope, envelope derivative, syllable rate (peakRate)   [3 cols]
  spectral family : f0, log-mel PC1-4                                          [5 cols]
  visual family   : motion energy, spatial contrast                            [2 cols]

Requires: source/ from ft_ica_full3.m; the clip files.
"""
import os, numpy as np
from lib import loo, resid, build_features, load_source, region_groups, _z as z
ROOT = r"G:/Barak1"
DERIV = os.path.join(ROOT, "reanalysis")   # reconstruction outputs and intermediate results; point anywhere

BETA = 3; RNG = np.random.RandomState(0)
SOURCE = os.path.join(DERIV, "source_myica3")          # objective-ICA region time courses
CACHE = r"G:/Barak1/paper_code/_feat_cache.npz"
CONDS = ["Charismatic", "Non_Charismatic"]
FAMILIES = {"envelope": [0, 1, 2], "spectral": [3, 4, 5, 6, 7], "visual": [8, 9]}

if os.path.exists(CACHE):
    _c = np.load(CACHE); F = {c: _c[c] for c in CONDS}
else:
    F = {c: build_features(c) for c in CONDS}
    np.savez(CACHE, **F)

S = {c: load_source(c, folder=SOURCE) for c in CONDS}
T = {c: min(S[c].shape[2], len(F[c])) for c in CONDS}


def scramble(M):
    """Phase-scramble each column: identical magnitude spectrum, no true stimulus locking."""
    out = np.zeros_like(M)
    for k in range(M.shape[1]):
        Fk = np.fft.rfft(M[:, k]); ph = RNG.uniform(0, 2*np.pi, len(Fk)); ph[0] = 0
        out[:, k] = np.fft.irfft(np.abs(Fk)*np.exp(1j*ph), n=M.shape[0])
    return z(out)


NSCRAMBLE = 20      # a single phase-scramble draw varies by ~+/-20 percentage points; average it


def effect(design=None, scramble_it=False):
    """Charismatic - Non_Charismatic beta ISC per region, after regressing `design` (cols or None)."""
    iz = {}
    for c in CONDS:
        X = S[c][:, :, :T[c], BETA]
        if design is not None:
            M = F[c][:T[c]][:, design]
            M = scramble(M) if scramble_it else M
            X = np.stack([resid(X[:, r], M) for r in range(72)], 1)
        iz[c] = loo(X)
    return (iz["Charismatic"] - iz["Non_Charismatic"]).mean(0)


def null_mean(cols):
    """Mean retention over NSCRAMBLE independent phase-scramble draws (single draws are noisy)."""
    return np.mean([effect(cols, scramble_it=True) for _ in range(NSCRAMBLE)], axis=0)

raw = effect()
res = {f: effect(cols) for f, cols in FAMILIES.items()}
nul = {f: null_mean(cols) for f, cols in FAMILIES.items()}
res["all 10"] = effect(sum(FAMILIES.values(), [])); nul["all 10"] = null_mean(sum(FAMILIES.values(), []))
order = ["envelope", "spectral", "visual", "all 10"]

gi = {g: i for g, i in region_groups().items() if i}
pct = lambda v, r0: 100*v/r0 if r0 else float("nan")

print("Beta ISC condition effect retained after regressing each feature family")
print("(real / mean of 20 phase-scrambled nulls, % of the raw effect)\n")
print(f"  {'group':11s} | {'raw':>7s} | " + " | ".join(f"{f:>17s}" for f in order))
for g, idx in gi.items():
    r0 = raw[idx].mean()
    cells = " | ".join(f"{pct(res[f][idx].mean(), r0):6.0f}% /{pct(nul[f][idx].mean(), r0):5.0f}%" for f in order)
    print(f"  {g:11s} | {r0:+.4f} | {cells}")
r0 = raw.mean()
cells = " | ".join(f"{pct(res[f].mean(), r0):6.0f}% /{pct(nul[f].mean(), r0):5.0f}%" for f in order)
print(f"  {'WHOLE BRAIN':11s} | {r0:+.4f} | {cells}")

print("\nRead: a family is doing real work only where its real retention is well BELOW its own")
print("scrambled null. Equal values mean the family removed nothing beyond spent d.o.f.")

# ---- incremental test: does the envelope carry anything the spectral features do not? ----
SPEC = FAMILIES["spectral"]; ENV = FAMILIES["envelope"]
inc = {
    "envelope alone (1 col)":  effect([0]),
    "envelope family":         effect(ENV),
    "spectral family":         effect(SPEC),
    "spectral + envelope":     effect(SPEC + ENV),
    "all 10":                  effect(sum(FAMILIES.values(), [])),
}
print("\n\nIncremental test (whole brain): what does the envelope add over the spectral features?\n")
print(f"  {'design':24s} | {'n cols':>6s} | {'retained':>8s}")
ncols = {"envelope alone (1 col)": 1, "envelope family": 3, "spectral family": 5,
         "spectral + envelope": 8, "all 10": 10}
for k, v in inc.items():
    print(f"  {k:24s} | {ncols[k]:6d} | {pct(v.mean(), raw.mean()):7.0f}%")
d = pct(inc["spectral + envelope"].mean(), raw.mean()) - pct(inc["spectral family"].mean(), raw.mean())
print(f"\n  Envelope family's unique contribution over the spectral features: {-d:+.1f} percentage points")
