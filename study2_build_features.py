"""Build the Study-1 reviewer feature battery for the Studies 2/3 stimuli, aligned to the MEG.

Features (10, same as study1_isc.py):
  acoustic: envelope, env_deriv, f0, syllable_rate(peakRate), mel_PC1..4
  visual:   motion energy, spatial RMS contrast

Stimuli:  S2 -> Video.wmv (516.2 s)
          S3 -> Video2.wmv (no-cue) and Video2Heads.wmv (cue)  [audio identical, video differs]

Alignment: the audio starts ONSET seconds AFTER the MEG epoch start; the lag was estimated
by cross-correlating the acoustic envelope against the MEG, blind to group.
So neural sample n (at 100 Hz) sees feature at audio index n - ONSET*100 -> prepend ONSET*100 pad.

Caches to features/<tag>_feat100.npy  (T x 10, z-scored, 0.8 Hz low-passed, MEG-aligned).
"""
import os, subprocess, numpy as np, soundfile as sf, librosa, scipy.io as sio
from scipy.signal import hilbert, butter, filtfilt, resample_poly, find_peaks

ROOT = "G:/Barak1/iscex"; FS = 100
OUT = os.path.join(ROOT, "features"); os.makedirs(OUT, exist_ok=True)
try:
    import imageio_ffmpeg; FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG = r"C:/Users/Avi/.stacher/ffmpeg.exe"
NAMES = ["envelope","env_deriv","f0","syllable_rate","mel_PC1","mel_PC2","mel_PC3","mel_PC4","motion","contrast"]
ONSET = {"S2": 1.02, "S3": 0.92}          # seconds; group-blind estimate, see docstring

def lp08(x): b,a = butter(4, 0.8/(FS/2), "low"); return filtfilt(b, a, x, axis=0)
def z(x):    return (x - x.mean(0)) / (x.std(0) + 1e-9)

def build(video, T, onset):
    wav = os.path.join(OUT, "_tmp.wav")
    subprocess.run([FFMPEG,"-y","-loglevel","error","-i",os.path.join(ROOT,video),
                    "-vn","-ac","1","-ar","16000","-f","wav",wav], check=True)
    y, sr = sf.read(wav); os.remove(wav); y = y.astype(float)
    env = np.abs(hilbert(y)); bb,aa = butter(4, 30/(sr/2), "low"); env = filtfilt(bb,aa,env)
    e = resample_poly(env, FS, sr); d = np.diff(e, prepend=e[0]); dr = d.copy(); dr[dr<0]=0
    pk,_ = find_peaks(dr); syl = np.zeros_like(dr); syl[pk] = dr[pk]
    print(f"  {video}: f0 (pyin, slow)...", flush=True)
    f0,_,_ = librosa.pyin(y, fmin=75, fmax=350, sr=sr, frame_length=2048, hop_length=160)
    f0 = np.nan_to_num(f0, nan=np.nanmedian(f0)); f0 = resample_poly(f0, FS, 100)
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=32, hop_length=160))
    melpc = resample_poly(z(mel.T) @ np.linalg.svd(z(mel.T), full_matrices=False)[2][:4].T, FS, 100, axis=0)
    W,H,FPS = 160,128,25
    print(f"  {video}: decoding video...", flush=True)
    rv = subprocess.run([FFMPEG,"-loglevel","error","-i",os.path.join(ROOT,video),
        "-vf",f"scale={W}:{H},format=gray,fps={FPS}","-f","rawvideo","-pix_fmt","gray","pipe:1"],
        capture_output=True).stdout
    nf = len(rv)//(W*H); Fv = np.frombuffer(rv[:nf*W*H], np.uint8).reshape(nf,H,W).astype(float)/255
    motion = resample_poly(np.concatenate([[0], np.abs(np.diff(Fv,axis=0)).mean((1,2))]), FS, FPS)
    contrast = resample_poly(Fv.std((1,2)), FS, FPS)
    cols = [e,d,f0,syl,melpc[:,0],melpc[:,1],melpc[:,2],melpc[:,3],motion,contrast]
    pad = int(round(onset*FS))
    out = []
    for c in cols:
        c = np.concatenate([np.full(pad, c[0]), c])            # audio starts `onset` s after epoch
        c = np.pad(c, (0, max(0, T-len(c))), "edge")[:T]
        out.append(lp08(c))
    return z(np.column_stack(out))

JOBS = [("S2",       "Video.wmv",        51574, ONSET["S2"]),   # T = S2_band n samples @100Hz
        ("S3_nocue", "Video2.wmv",       35666, ONSET["S3"]),
        ("S3_cue",   "Video2Heads.wmv",  35666, ONSET["S3"])]
if __name__ == "__main__":
    for tag, vid, T, on in JOBS:
        f = os.path.join(OUT, f"{tag}_feat100.npy")
        if os.path.exists(f): print(f"{tag}: cached"); continue
        F = build(vid, T, on); np.save(f, F.astype(np.float32))
        cc = np.corrcoef(F.T); mx = np.abs(cc - np.eye(len(NAMES))).max()
        print(f"{tag}: {F.shape} saved; max |r| off-diag = {mx:.2f}")
    # sanity: S3 cue vs nocue should differ ONLY in the visual columns
    a = np.load(os.path.join(OUT,"S3_nocue_feat100.npy")); b = np.load(os.path.join(OUT,"S3_cue_feat100.npy"))
    for i,n in enumerate(NAMES):
        print(f"  S3 cue-vs-nocue {n:14s} r={np.corrcoef(a[:,i],b[:,i])[0,1]:+.4f}")
