"""Study 1: does a FINER visual model remove the surviving OCCIPITAL beta ISC? (editor #7)

The feature battery used a coarse visual model (global motion energy, luminance, contrast).
Occipital beta ISC survived it (~80% retained), but that survival could reflect under-modelled visual
dynamics rather than a genuine non-visual effect. Here we build a richer visual model (numpy+ndimage,
no cv2) and test whether it removes more occipital beta ISC than the coarse model — controlling for the
added degrees of freedom with a phase-scrambled null.

Fine visual features (per frame, 25 fps -> 100 Hz, 0.8 Hz LP, z):
  motE       global motion energy (|frame diff|)                 [also in coarse]
  normalflow optic-flow proxy = mean |I_t| / |grad I|  (normal flow, gradient-normalised motion)
  mot_center motion in the central (speaker/face) region
  mot_coarse low-spatial-frequency motion (4x block-downsampled frame diff)
  flicker    |temporal change of mean luminance|
  hi_sf      high-spatial-frequency energy (variance of Laplacian)
  grad_en    edge energy (mean Sobel gradient magnitude)
  lum        luminance                                            [also in coarse]
  contrast   spatial RMS contrast                                 [also in coarse]

Models compared (each regressed out of every region, per band; occipital/temporal beta focus):
  raw  |  acoustic-only(8)  |  +coarse-visual(3)  |  +fine-visual(9)  |  +coarse-null  |  +fine-null

Usage: python study1_visual_fine_control.py
"""
import os, glob, re, sys, subprocess, numpy as np, soundfile as sf, librosa, scipy.io as sio
from scipy import ndimage, stats
from scipy.signal import hilbert, butter, filtfilt, resample_poly, find_peaks
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
NSCRAMBLE = 20     # single-draw phase-scramble nulls vary by ~+/-20 percentage points
from isc_source import loo, fdr, resid_region, scramble, z, lp08, labels, GROUPS, CLIP, DUR, FS, SRC
import imageio_ffmpeg; FFMPEG = imageio_ffmpeg.get_ffmpeg_exe(); ROOT = r"G:/Barak1"
ROOT = r"G:/Barak1"
DERIV = os.path.join(ROOT, "reanalysis")   # reconstruction outputs and intermediate results; point anywhere
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]

def fit_col(c, T):                                   # pad/truncate -> 0.8 Hz LP  (z applied per-matrix later)
    return lp08(np.pad(c, (0, max(0, T - len(c))), "edge")[:T])

def pca(X, k):                                       # top-k PCs of a feature block, z-scored
    Xc = X - X.mean(0); U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    return z(U[:, :k] * S[:k])

def build_acoustic(clip, T):
    wav = os.path.join(SRC, "_v.wav")
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", os.path.join(ROOT, clip),
                    "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", wav], check=True)
    y, sr = sf.read(wav); os.remove(wav); y = y.astype(float)
    env = np.abs(hilbert(y)); bb, aa = butter(4, 30/(sr/2), "low"); env = filtfilt(bb, aa, env)
    e = resample_poly(env, FS, sr); d = np.diff(e, prepend=e[0]); dr = d.copy(); dr[dr < 0] = 0
    pk, _ = find_peaks(dr); syl = np.zeros_like(dr); syl[pk] = dr[pk]
    f0, _, _ = librosa.pyin(y, fmin=75, fmax=350, sr=sr, frame_length=2048, hop_length=160)
    f0 = np.nan_to_num(f0, nan=np.nanmedian(f0)); f0 = resample_poly(f0, FS, 100)
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=32, hop_length=160))
    melpc = resample_poly(z(mel.T) @ np.linalg.svd(z(mel.T), full_matrices=False)[2][:4].T, FS, 100, axis=0)
    cols = [e, d, f0, syl, melpc[:, 0], melpc[:, 1], melpc[:, 2], melpc[:, 3]]
    return z(np.column_stack([fit_col(c, T) for c in cols]))

def read_frames(clip, W=192, H=144, FPS=25):
    rv = subprocess.run([FFMPEG, "-loglevel", "error", "-i", os.path.join(ROOT, clip),
        "-vf", f"scale={W}:{H},format=gray,fps={FPS}", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"],
        capture_output=True).stdout
    nf = len(rv) // (W*H)
    return np.frombuffer(rv[:nf*W*H], np.uint8).reshape(nf, H, W), FPS

def build_visual(clip, T):
    Fv, FPS = read_frames(clip); nf, H, W = Fv.shape
    cy0, cy1, cx0, cx1 = H//4, 3*H//4, W//4, 3*W//4
    hb, wb = H//4, W//4
    motE = np.zeros(nf); nflow = np.zeros(nf); motC = np.zeros(nf); motCo = np.zeros(nf)
    flick = np.zeros(nf); hisf = np.zeros(nf); grad = np.zeros(nf); lum = np.zeros(nf); con = np.zeros(nf)
    prev = None; plum = None
    for f in range(nf):
        fr = Fv[f].astype(np.float32) / 255.0
        lum[f] = fr.mean(); con[f] = fr.std()
        gx = ndimage.sobel(fr, axis=1); gy = ndimage.sobel(fr, axis=0); gmag = np.hypot(gx, gy)
        grad[f] = gmag.mean(); hisf[f] = ndimage.laplace(fr).var()
        if prev is not None:
            dt = np.abs(fr - prev); motE[f] = dt.mean()
            motC[f] = dt[cy0:cy1, cx0:cx1].mean()
            nflow[f] = np.mean(dt / (gmag + 0.05))                         # normal-flow proxy
            frc = fr.reshape(hb, 4, wb, 4).mean((1, 3)); prc = prev.reshape(hb, 4, wb, 4).mean((1, 3))
            motCo[f] = np.abs(frc - prc).mean()                            # low-spatial-freq motion
            flick[f] = abs(lum[f] - plum)
        prev = fr; plum = lum[f]
    cols = [motE, nflow, motC, motCo, flick, hisf, grad, lum, con]
    names = ["motE", "normalflow", "mot_center", "mot_coarse", "flicker", "hi_sf", "grad_en", "lum", "contrast"]
    V = z(np.column_stack([fit_col(resample_poly(c, FS, FPS), T) for c in cols]))
    return V, names

def diff_after(dat, Mc, Mn):
    res = {}
    for bi, b in enumerate(BANDS):
        C, N = dat["Charismatic"][:, :, :, bi], dat["Non_Charismatic"][:, :, :, bi]
        if Mc is not None:
            C = np.stack([resid_region(C[:, r], Mc) for r in range(72)], 1)
            N = np.stack([resid_region(N[:, r], Mn) for r in range(72)], 1)
        iC, iN = loo(C), loo(N)
        res[b] = dict(d=(iC - iN).mean(0), sig=fdr(stats.ttest_rel(iC, iN, alternative="greater")[1]))
    return res

def main():
    subs = sorted([p for p in glob.glob(os.path.join(SRC, "char_*.mat"))
                   if re.fullmatch(r"char_\d+", os.path.basename(p)[:-4])],
                  key=lambda p: int(os.path.basename(p)[5:-4]))
    print(f"{len(subs)} source files", flush=True)
    T = {c: int(DUR[c] * FS) for c in DUR}
    dat = {c: np.stack([sio.loadmat(s)[c].astype(np.float32) for s in subs]) for c in DUR}
    AC, VF, names = {}, {}, None
    for c in DUR:
        AC[c] = build_acoustic(CLIP[c], T[c])
        VF[c], names = build_visual(CLIP[c], T[c])
    VCi = [names.index(x) for x in ["motE", "lum", "contrast"]]           # coarse = existing 3
    VC = {c: VF[c][:, VCi] for c in DUR}
    print(f"fine visual = {len(names)} features: {names}", flush=True)
    for c in DUR:
        rr = np.corrcoef(VF[c][:, names.index("normalflow")], VF[c][:, names.index("motE")])[0, 1]
        print(f"  {c:16s}: r(normalflow, motE)={rr:+.2f}", flush=True)

    VP = {c: pca(VF[c], 4) for c in DUR}                                  # rich visual -> top-4 PCs (denoise, dof-match)
    ev = np.linalg.svd(VF["Charismatic"] - VF["Charismatic"].mean(0), compute_uv=False)**2
    print(f"  fine-visual top-4 PCs explain {100*ev[:4].sum()/ev.sum():.0f}% of the 9-feature variance", flush=True)
    M = {c: {} for c in DUR}
    for c in DUR:
        M[c]["ac"] = AC[c]
        M[c]["coarse"] = np.column_stack([AC[c], VC[c]])
        M[c]["fine"] = np.column_stack([AC[c], VP[c]])
        M[c]["coarse_null"] = np.column_stack([AC[c], scramble(VC[c])])
        M[c]["fine_null"] = np.column_stack([AC[c], scramble(VP[c])])
        for j in range(1, NSCRAMBLE):        # extra draws, averaged below
            M[c][f"coarse_null{j}"] = np.column_stack([AC[c], scramble(VC[c])])
            M[c][f"fine_null{j}"] = np.column_stack([AC[c], scramble(VP[c])])
    raw = diff_after(dat, None, None)
    R = {k: diff_after(dat, M["Charismatic"][k], M["Non_Charismatic"][k])
         for k in ["ac", "coarse", "fine"] +
             [f"{n}{j}" if j else n for n in ("coarse_null", "fine_null") for j in range(NSCRAMBLE)]}
    for n in ("coarse_null", "fine_null"):    # collapse the draws to their mean
        for b in BANDS:
            R[n][b]["d"] = np.mean([R[f"{n}{j}" if j else n][b]["d"] for j in range(NSCRAMBLE)], axis=0)

    print("\nFDR-sig C>NC regions per band:", flush=True)
    print("  band   raw | acoustic | +coarse | +fine | +coarse-null | +fine-null", flush=True)
    for b in BANDS:
        print(f"  {b:6s} {raw[b]['sig'].sum():3d} | {R['ac'][b]['sig'].sum():6d} | "
              f"{R['coarse'][b]['sig'].sum():5d} | {R['fine'][b]['sig'].sum():5d} | "
              f"{R['coarse_null'][b]['sig'].sum():10d} | {R['fine_null'][b]['sig'].sum():8d}", flush=True)

    gi = {g: [i for i in range(72) if any(k in labels[i] for k in ks)] for g, ks in GROUPS.items()}
    print("\n[beta] group ΔISC retained vs raw:  acoustic -> +coarse -> +fine  (+fine-null)", flush=True)
    for g, idx in gi.items():
        if not idx: continue
        dr = raw['beta']['d'][idx].mean()
        da = R['ac']['beta']['d'][idx].mean(); dc = R['coarse']['beta']['d'][idx].mean()
        df = R['fine']['beta']['d'][idx].mean(); dfn = R['fine_null']['beta']['d'][idx].mean()
        pc = lambda x: 100*x/dr if dr else 0
        print("  %-11s %2d | raw %+.4f | ac %3.0f%% -> coarse %3.0f%% -> fine %3.0f%%  (fine-null %3.0f%%)"
              % (g, len(idx), dr, pc(da), pc(dc), pc(df), pc(dfn)), flush=True)
    np.save(os.path.join(DERIV, "isc_source_visual_fine_results.npy"),
            dict(raw=raw, **R), allow_pickle=True)
    print("\nSaved isc_source_visual_fine_results.npy", flush=True)

if __name__ == "__main__":
    main()
