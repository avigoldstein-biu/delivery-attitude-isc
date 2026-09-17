"""
Shared functions for the Study 1 frequency-resolved ISC analysis.
Imported by the numbered analysis scripts. No side effects on import except reading
the AAL region labels (path in CONFIG below).
"""
import os, subprocess, numpy as np, soundfile as sf, librosa, scipy.io as sio
from scipy.signal import hilbert, butter, filtfilt, resample_poly, find_peaks
import imageio_ffmpeg

# ---------------------------------------------------------------- CONFIG (edit paths)
CLIP_DIR   = r"G:/Barak1"                                   # folder with the .MPG clips
SOURCE_DIR = r"G:/Barak1/reanalysis/source_myica3"         # objective-ICA region time courses (from ft_ica_full3.m)
BB_DIR     = r"G:/Barak1/reanalysis/source_bb_ica"         # broadband region time courses
ATLAS_INFO = r"G:/Barak1/Charisma/atlas_info.mat"          # AAL tissue map (1169 grid -> 72 regions)
FFMPEG     = imageio_ffmpeg.get_ffmpeg_exe()

FS = 100                                                    # ISC-signal sampling rate (Hz)
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 12), "beta": (12, 25), "gamma": (25, 40)}
CLIP = {"Charismatic": "Charismatic.MPG", "Non_Charismatic": "Non-Charismatic.MPG", "Silent": "Silent.MPG"}
DUR  = {"Charismatic": 150.24, "Non_Charismatic": 140.16, "Silent": 153.60}

labels = [str(x).strip("'") for x in sio.loadmat(ATLAS_INFO, squeeze_me=True)["tissuelabel"]]
GROUPS = {"occipital": ["Occipital", "Calcarine", "Lingual", "Cuneus"], "temporal": ["Temporal", "Heschl"],
          "frontal": ["Frontal", "Precentral", "Rolandic", "Rectus"],
          "parietal": ["Parietal", "Postcentral", "Precuneus", "Angular", "SupraMarginal"],
          "insula": ["Insula"], "cingulate": ["Cingul"]}
def region_groups():
    return {g: [i for i in range(len(labels)) if any(k in labels[i] for k in ks)] for g, ks in GROUPS.items()}

# ---------------------------------------------------------------- ISC + stats
def loo(arr):
    """Leave-one-out ISC. arr (N_subj, N_region, T) -> (N_subj, N_region) Fisher-z."""
    N = arr.shape[0]; tot = arr.sum(0); out = np.zeros((N, arr.shape[1]))
    for i in range(N):
        o = (tot-arr[i])/(N-1); a = arr[i]-arr[i].mean(-1, keepdims=True); oo = o-o.mean(-1, keepdims=True)
        r = (a*oo).sum(-1)/(np.sqrt((a**2).sum(-1)*(oo**2).sum(-1))+1e-12)
        out[i] = np.arctanh(np.clip(r, -.999, .999))
    return out

def fdr(p, q=0.05):
    """Benjamini-Hochberg FDR at q. Returns boolean significance mask."""
    p = np.asarray(p); o = np.argsort(p); n = len(p); thr = q*np.arange(1, n+1)/n
    below = p[o] <= thr; k = np.where(below)[0].max() if below.any() else -1
    sig = np.zeros(n, bool)
    if k >= 0: sig[o[:k+1]] = True
    return sig

def rm_anova(X):
    """One-way repeated-measures ANOVA. X (n_subj, n_cond) -> F, p, eta_p2, df1, df2."""
    from scipy import stats
    n, k = X.shape; gm = X.mean()
    ss_c = n*((X.mean(0)-gm)**2).sum(); ss_s = k*((X.mean(1)-gm)**2).sum()
    ss_e = ((X-gm)**2).sum()-ss_c-ss_s; d1, d2 = k-1, (n-1)*(k-1)
    F = (ss_c/d1)/(ss_e/d2)
    return F, stats.f.sf(F, d1, d2), ss_c/(ss_c+ss_e), d1, d2

# ---------------------------------------------------------------- feature regression
def pcs(F, k):
    """Top-k principal components of a feature matrix F (T, n_feat) -> (T, k)."""
    Fz = (F-F.mean(0))/(F.std(0)+1e-9); U, S, _ = np.linalg.svd(Fz, full_matrices=False); return U[:, :k]*S[:k]

def resid(P, M):
    """Regress design M (T,k) out of signals P (N_subj, T) along time."""
    Mp = np.linalg.pinv(M); return P - (P @ Mp.T) @ M.T

# ---------------------------------------------------------------- aperiodic (1/f) removal
_CF = np.array([2.5, 6.0, 10.0, 18.5, 32.5]); _X = np.column_stack([np.ones(5), np.log(_CF)]); _Xp = np.linalg.pinv(_X)
def periodic(dat):
    """Remove the aperiodic (1/f) component from the 5-band spectrum at each timepoint.
    dat (N,region,T,5) band amplitude -> periodic log-power residual (same shape)."""
    logP = 2*np.log(np.clip(dat, 1e-12, None))
    coef = np.einsum("ck,nrtk->nrtc", _Xp, logP)
    return logP - np.einsum("kc,nrtc->nrtk", _X, coef)

# ---------------------------------------------------------------- low-level stimulus features
def _lp08(x): b, a = butter(4, 0.8/(FS/2), "low"); return filtfilt(b, a, x, axis=0)
def _z(x): return (x - x.mean(0)) / (x.std(0) + 1e-9)

def build_features(cond):
    """Reviewer-named low-level feature matrix for a clip, at FS Hz, 0.8 Hz low-passed,
    z-scored: [envelope, env-derivative, syllable-rate(peakRate), f0, mel-PC1..4, motion,
    contrast]. Returns (T, 10)."""
    mpg = CLIP[cond]; T = int(DUR[cond]*FS); wav = os.path.join(CLIP_DIR, "_tmp_feat.wav")
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", os.path.join(CLIP_DIR, mpg),
                    "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", wav], check=True)
    y, sr = sf.read(wav); os.remove(wav); y = y.astype(float)
    env = np.abs(hilbert(y)); b, a = butter(4, 30/(sr/2), "low"); env = filtfilt(b, a, env)
    e = resample_poly(env, FS, sr); d = np.diff(e, prepend=e[0]); dr = d.copy(); dr[dr < 0] = 0
    pk, _ = find_peaks(dr); syl = np.zeros_like(dr); syl[pk] = dr[pk]
    f0, _, _ = librosa.pyin(y, fmin=75, fmax=350, sr=sr, frame_length=2048, hop_length=160)
    f0 = np.nan_to_num(f0, nan=np.nanmedian(f0)); f0 = resample_poly(f0, FS, 100)
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=32, hop_length=160))
    melpc = resample_poly(_z(mel.T) @ np.linalg.svd(_z(mel.T), full_matrices=False)[2][:4].T, FS, 100, axis=0)
    W, H, FPS = 160, 128, 25
    rv = subprocess.run([FFMPEG, "-loglevel", "error", "-i", os.path.join(CLIP_DIR, mpg),
        "-vf", f"scale={W}:{H},format=gray,fps={FPS}", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"], capture_output=True).stdout
    nf = len(rv)//(W*H); Fv = np.frombuffer(rv[:nf*W*H], np.uint8).reshape(nf, H, W).astype(float)/255
    motion = resample_poly(np.concatenate([[0], np.abs(np.diff(Fv, axis=0)).mean((1, 2))]), FS, FPS)
    contrast = resample_poly(Fv.std((1, 2)), FS, FPS)
    cols = [e, d, syl, f0, melpc[:, 0], melpc[:, 1], melpc[:, 2], melpc[:, 3], motion, contrast]
    return _z(np.column_stack([_lp08(np.pad(c, (0, max(0, T-len(c))), "edge")[:T]) for c in cols]))

def load_source(cond, folder=None):
    """Load objective-ICA region band-power ISC signals for a condition.
    Returns (N_subj, 72, T, 5)."""
    import glob
    folder = folder or SOURCE_DIR
    subs = sorted([s for s in glob.glob(os.path.join(folder, "char_*.mat")) if "_" not in os.path.basename(s)[5:-4]],
                  key=lambda p: int(os.path.basename(p)[5:-4]))
    A = [sio.loadmat(s)[cond] for s in subs]; T = min(a.shape[1] for a in A)
    return np.stack([a[:, :T].astype(np.float32) for a in A])
