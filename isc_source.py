"""
Shared constants and helpers for the Study 1 source-space analyses.

Imported by study1_visual_fine_control.py, study1_power_specparam.py,
study1_brain_behavior.py and study1_is_rsa.py. Not a script -- the analyses live in the
study1_*.py files.

SRC points at the objective-ICA reconstruction written by matlab/ft_ica_full3.m: 72 AAL
regions x time at 100 Hz x 5 bands, one file per participant, holding all three conditions.
"""
import os
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt
ROOT = r"G:/Barak1"
DERIV = os.path.join(ROOT, "reanalysis")   # reconstruction outputs and intermediate results; point anywhere

SRC = os.path.join(DERIV, "source_myica3")   # from matlab/ft_ica_full3.m
FS = 100
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
CLIP = {"Charismatic": "Charismatic.MPG", "Non_Charismatic": "Non-Charismatic.MPG"}
DUR = {"Charismatic": 150.24, "Non_Charismatic": 140.16}
labels = [str(x).strip("'") for x in
          sio.loadmat(os.path.join(ROOT, "Charisma", "atlas_info.mat"), squeeze_me=True)["tissuelabel"]]
GROUPS = {"occipital": ["Occipital", "Calcarine", "Lingual", "Cuneus"], "temporal": ["Temporal", "Heschl"],
          "frontal": ["Frontal", "Precentral", "Rolandic", "Rectus"],
          "parietal": ["Parietal", "Postcentral", "Precuneus", "Angular", "SupraMarginal"],
          "insula": ["Insula"], "cingulate": ["Cingul"]}
rng = np.random.RandomState(0)


def lp08(x): b, a = butter(4, 0.8/(FS/2), "low"); return filtfilt(b, a, x, axis=0)
def z(x): return (x - x.mean(0)) / (x.std(0) + 1e-9)


def loo(arr):                                          # arr (N, R, T) -> (N, R) Fisher-z ISC
    N = arr.shape[0]; tot = arr.sum(0); out = np.zeros((N, arr.shape[1]))
    for i in range(N):
        o = (tot-arr[i])/(N-1); a = arr[i]-arr[i].mean(-1, keepdims=True); oo = o-o.mean(-1, keepdims=True)
        r = (a*oo).sum(-1)/(np.sqrt((a**2).sum(-1)*(oo**2).sum(-1))+1e-12)
        out[i] = np.arctanh(np.clip(r, -.999, .999))
    return out


def fdr(p, q=0.05):
    p = np.asarray(p); o = np.argsort(p); n = len(p); thr = q*(np.arange(1, n+1))/n
    below = p[o] <= thr; k = np.where(below)[0].max() if below.any() else -1
    sig = np.zeros(n, bool); sig[o[:k+1]] = k >= 0; return sig


def resid_region(X, M):                                # X (N,T) region across subjects; regress M (T,k)
    Xp = np.linalg.pinv(M); return X - (X @ Xp.T) @ M.T


def scramble(M):                                       # phase-scramble each feature (matched complexity null)
    out = np.zeros_like(M)
    for k in range(M.shape[1]):
        X = np.fft.rfft(M[:, k]); ph = rng.uniform(0, 2*np.pi, len(X)); ph[0] = 0
        out[:, k] = np.fft.irfft(np.abs(X)*np.exp(1j*ph), n=M.shape[0])
    return z(out)
