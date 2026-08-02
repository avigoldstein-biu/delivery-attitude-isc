"""
Study 1: band-resolved ISC for the charismatic vs non-charismatic contrast, at each stage of
low-level control. Uses the full ten-regressor feature matrix.

Covers the two speech conditions; the three-condition comparison against silence is in
study1_anova.py.

Stages, per band:
  raw      uncontrolled
  feat     + low-level feature regression (full 10-column matrix, per condition)
  nul      + phase-scrambled feature matrix (matched spectrum and column count)
  both     + feature regression on the aperiodic-removed (periodic) component
and for beta additionally:
  feat_w   feature matrix + word-onset regressor (forced alignment)
  feat_wn  feature matrix + phase-scrambled word-onset regressor

Writes study1_ica_results.npz and prints the numbers the manuscript and supplement cite.
"""
import os, sys, glob, json, gc, numpy as np, scipy.io as sio
from scipy import stats
from scipy.signal import butter, filtfilt
ROOT = r"G:/Barak1"
DERIV = os.path.join(ROOT, "reanalysis")   # reconstruction outputs and intermediate results; point anywhere

_SET = sys.argv[1] if len(sys.argv) > 1 else "source_myica"
_ALT = "two-sided" if len(sys.argv) < 3 or sys.argv[2] == "two-sided" else "greater"
_NPC = int(sys.argv[3]) if len(sys.argv) > 3 else 0     # 0 = full matrix; else top-N feature PCs
SRC = os.path.join(DERIV, _SET)
ASR = os.path.join(DERIV, "asr")                    # forced-alignment word onsets
FEAT = os.path.join(ROOT, "paper_code", "_feat_cache.npz")       # 10-column battery, per condition
OUT = os.path.join(ROOT, "paper_code", f"study1_{_SET}_{_ALT}_pc{_NPC}.npz")
CONDS = ["Charismatic", "Non_Charismatic"]
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
BETA = 3; FS = 100
RNG = np.random.RandomState(0)

labels = [str(x).strip("'") for x in
          sio.loadmat(os.path.join(ROOT, "Charisma", "atlas_info.mat"), squeeze_me=True)["tissuelabel"]]
GROUPS = {"occipital": ["Occipital", "Calcarine", "Lingual", "Cuneus"],
          "temporal": ["Temporal", "Heschl"],
          "frontal": ["Frontal", "Precentral", "Rolandic", "Rectus"],
          "parietal": ["Parietal", "Postcentral", "Precuneus", "Angular", "SupraMarginal"],
          "insula": ["Insula"], "cingulate": ["Cingul"]}
ANAT = ["Rolandic_Oper_R", "Temporal_Sup_R", "Temporal_Mid_R", "Insula_R"]

_CF = np.array([2.5, 6.0, 10.0, 18.5, 32.5], np.float32)
_X = np.column_stack([np.ones(5, np.float32), np.log(_CF)]).astype(np.float32)
_Xp = np.linalg.pinv(_X).astype(np.float32)


def periodic(dat):
    """Remove the aperiodic (1/f) component across the 5-band spectrum at each timepoint."""
    logP = (2 * np.log(np.clip(dat, 1e-12, None))).astype(np.float32)
    coef = np.einsum("ck,nrtk->nrtc", _Xp, logP, optimize=True)
    out = logP - np.einsum("kc,nrtc->nrtk", _X, coef, optimize=True)
    del logP, coef
    return out


def loo(arr):
    N = arr.shape[0]; tot = arr.sum(0); out = np.zeros((N, arr.shape[1]))
    for i in range(N):
        o = (tot - arr[i]) / (N - 1)
        a = arr[i] - arr[i].mean(-1, keepdims=True); oo = o - o.mean(-1, keepdims=True)
        r = (a * oo).sum(-1) / (np.sqrt((a ** 2).sum(-1) * (oo ** 2).sum(-1)) + 1e-12)
        out[i] = np.arctanh(np.clip(r, -.999, .999))
    return out


def fdr(p, q=0.05):
    p = np.asarray(p); o = np.argsort(p); n = len(p); thr = q * np.arange(1, n + 1) / n
    below = p[o] <= thr; k = np.where(below)[0].max() if below.any() else -1
    sig = np.zeros(n, bool)
    if k >= 0: sig[o[:k + 1]] = True
    return sig


def resid_all(P, M):
    """Regress design M (T,k) out of every region of P (N,72,T)."""
    Mp = np.linalg.pinv(M)
    return np.stack([P[:, r] - (P[:, r] @ Mp.T) @ M.T for r in range(P.shape[1])], 1)


def scramble(M):
    out = np.zeros_like(M)
    for k in range(M.shape[1]):
        Fk = np.fft.rfft(M[:, k]); ph = RNG.uniform(0, 2 * np.pi, len(Fk)); ph[0] = 0
        out[:, k] = np.fft.irfft(np.abs(Fk) * np.exp(1j * ph), n=M.shape[0])
    return (out - out.mean(0)) / (out.std(0) + 1e-9)


def word_onsets(cond, T):
    js = json.load(open(os.path.join(ASR, f"{cond}_words.json"), encoding="utf-8"))
    on = np.zeros(T)
    for w in js["words"]:
        i = int(round(w["start"] * FS))
        if 0 <= i < T: on[i] = 1.0
    b, a = butter(4, 0.8 / (FS / 2), "low"); on = filtfilt(b, a, on)
    return ((on - on.mean()) / (on.std() + 1e-9))[:, None]


def load(cond):
    import re
    subs = sorted([p for p in glob.glob(os.path.join(SRC, "char_*.mat"))
                   if re.fullmatch(r"char_\d+", os.path.basename(p)[:-4])],   # exclude *_silent.mat
                  key=lambda p: int(os.path.basename(p)[5:-4]))
    A = [sio.loadmat(s)[cond] for s in subs]; T = min(a.shape[1] for a in A)
    return np.stack([a[:, :T].astype(np.float32) for a in A])


# ---------------------------------------------------------------- compute
_f = np.load(FEAT)
iz = {c: {} for c in CONDS}
for cond in CONDS:
    A = load(cond); T = min(A.shape[2], len(_f[cond])); A = A[:, :, :T]
    F = _f[cond][:T]
    if _NPC:                                              # reduce to top-N PCs (limits over-stripping)
        Fz = (F - F.mean(0)) / (F.std(0) + 1e-9)
        U, S_, _ = np.linalg.svd(Fz, full_matrices=False); F = U[:, :_NPC] * S_[:_NPC]
    Fs = scramble(F); W = word_onsets(cond, T)
    per = periodic(A)
    print(f"{cond}: {A.shape[0]} subjects, T={T}")
    for bi, b in enumerate(BANDS):
        raw = A[:, :, :, bi]
        iz[cond][b] = {
            "raw":  loo(raw),
            "feat": loo(resid_all(raw, F)),
            "nul":  loo(resid_all(raw, Fs)),
            "both": loo(resid_all(per[:, :, :, bi], F)),
        }
        if b == "beta":
            iz[cond][b]["feat_w"] = loo(resid_all(raw, np.column_stack([F, W])))
            iz[cond][b]["feat_wn"] = loo(resid_all(raw, np.column_stack([F, scramble(W)])))
    del A, per; gc.collect()

# ---------------------------------------------------------------- contrasts
res = {}
for b in BANDS:
    for st in iz["Charismatic"][b]:
        a, c = iz["Charismatic"][b][st], iz["Non_Charismatic"][b][st]
        n = min(len(a), len(c))
        res[f"{b}_{st}_d"] = (a[:n] - c[:n]).mean(0)
        res[f"{b}_{st}_sig"] = fdr((stats.ttest_rel(a[:n], c[:n])[1] if _ALT == "two-sided" else stats.ttest_rel(a[:n], c[:n], alternative="greater")[1]))
np.savez(OUT, labels=np.array(labels), **res)

# ---------------------------------------------------------------- report
print("\nFDR-significant regions (Charismatic > Non_Charismatic), by band and stage:")
print(f"  {'band':7s} " + " ".join(f"{s:>8s}" for s in ["raw", "feat", "nul", "both"]))
for b in BANDS:
    print(f"  {b:7s} " + " ".join(f"{int(res[f'{b}_{s}_sig'].sum()):8d}" for s in ["raw", "feat", "nul", "both"]))

gi = {g: [i for i in range(72) if any(k in labels[i] for k in ks)] for g, ks in GROUPS.items()}
raw, feat, nul = res["beta_raw_d"], res["beta_feat_d"], res["beta_nul_d"]
print("\nBeta effect retained after feature regression, by region group (real / scrambled null):")
for g, idx in gi.items():
    r0 = raw[idx].mean()
    print(f"  {g:11s} raw={r0:+.4f}  real={100*feat[idx].mean()/r0:5.0f}%  null={100*nul[idx].mean()/r0:5.0f}%")
print(f"  {'WHOLE BRAIN':11s} raw={raw.mean():+.4f}  real={100*feat.mean()/raw.mean():5.0f}%  "
      f"null={100*nul.mean()/raw.mean():5.0f}%")

core = np.where(res["beta_both_sig"])[0]
print(f"\nCore surviving BOTH controls ({len(core)} regions):")
for i in core:
    print(f"  {labels[i]:22s} raw={raw[i]:+.4f}  +feat={feat[i]:+.4f}  +both={res['beta_both_d'][i]:+.4f}")

print("\nWord-onset control (beta, whole brain):")
for st, lab_ in [("raw", "uncontrolled"), ("feat", "10-feature battery"),
                 ("feat_w", "+ word onsets"), ("feat_wn", "+ scrambled words")]:
    print(f"  {lab_:22s} d={res[f'beta_{st}_d'].mean():+.4f}  n={int(res[f'beta_{st}_sig'].sum()):2d}")

print("\nLaterality within the a-priori right auditory-perisylvian ROI [(R-L)/(R+L)]:")
base = [r.rsplit("_", 1)[0] for r in ANAT]
for st, lab_ in [("raw", "beta raw"), ("both", "beta after both controls")]:
    d = res[f"beta_{st}_d"]
    R = np.array([d[labels.index(x + "_R")] for x in base]); L = np.array([d[labels.index(x + "_L")] for x in base])
    print(f"  {lab_:26s} R={R.mean():+.4f} L={L.mean():+.4f}  LI={(R.sum()-L.sum())/(R.sum()+L.sum()):+.3f}  "
          f"R>L {int((R > L).sum())}/4")
print(f"\nSaved {OUT}")
