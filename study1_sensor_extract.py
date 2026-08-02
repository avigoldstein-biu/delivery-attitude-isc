"""
One MEG pass per subject x condition (Charismatic, Non-Charismatic), feeding
reanalyses B (envelope coupling), C (delta band), D (FOOOF).

Clip identity + location come from audiodata.mat
(trialinfo 220=Charismatic, 240=Non-Charismatic).

Per subject x condition it saves:
  psd_freqs, psd (channel-averaged Welch power spectrum, for FOOOF)
  bandpow: delta/theta/alpha/beta/gamma (incl. DELTA, which the paper omitted)
  env_track_r : peak envelope-tracking corr over auditory ROI (correct alignment)
  slowpow_env_r : corr(acoustic envelope, theta power-envelope lowpassed 0.8 Hz)
                  
Output: reanalysis/meg_extract.npz
"""
import os, glob, subprocess, numpy as np, soundfile as sf, scipy.io as sio, mne
from scipy.signal import hilbert, butter, filtfilt, resample_poly, correlate, welch
import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
SF_MEG, FS = 1017.25, 250
ANCHOR_OFFSET = -1.30
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 12), "beta": (12, 25), "gamma": (25, 40)}
CODEMAP = {220: "Charismatic.MPG", 240: "Non-Charismatic.MPG"}

def clip_envelope(mpg, out_fs):
    wav = os.path.join(OUT, "_tmpm.wav")
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", os.path.join(ROOT, mpg),
                    "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", wav], check=True)
    y, sr = sf.read(wav); os.remove(wav)
    e = np.abs(hilbert(y.astype(float))); b, a = butter(4, 30/(sr/2), "low"); e = filtfilt(b, a, e)
    return resample_poly(e, out_fs, sr), len(y)/sr

def bp(v, lo, hi, fs):
    b, a = butter(4, [lo/(fs/2), hi/(fs/2)], "band"); return filtfilt(b, a, v, axis=-1)

ENV = {c: {"250": clip_envelope(c, 250), "100": clip_envelope(c, 100)} for c in CODEMAP.values()}

def find_meg(subj):
    for f in os.listdir(subj):
        if f.endswith("lf_c,rfhp0.1Hz") and "hb" in f and "xc" in f:
            return os.path.join(subj, f)

subs = sorted(glob.glob(os.path.join(SUBJ, "char_*")), key=lambda p: int(p.split("_")[-1]))
import sys
ROOT = r"G:/Barak1"
DERIV = os.path.join(ROOT, "reanalysis")   # reconstruction outputs and intermediate results; point anywhere
SUBJ = os.path.join(ROOT, "Charisma"); OUT = DERIV
if len(sys.argv) > 1: subs = subs[:int(sys.argv[1])]
store = {c: {"psd": [], "bandpow": [], "env_track_r": [], "slowpow_env_r": [],
             "env_track_r_null": [], "slowpow_env_r_null": [], "subj": []}
         for c in ["Charismatic", "Non-Charismatic"]}
psd_freqs = None

for sd in subs:
    tag = os.path.basename(sd); meg = find_meg(sd); ad = os.path.join(sd, "audiodata.mat")
    if not meg or not os.path.exists(ad):
        continue
    m = sio.loadmat(ad, squeeze_me=True, struct_as_record=False)["audiodata"]
    ti = np.atleast_1d(m.trialinfo); si = np.atleast_2d(m.sampleinfo)
    raw = mne.io.read_raw_bti(meg, config_fname=os.path.join(sd, "config"),
        head_shape_fname=os.path.join(sd, "hs_file"), preload=False, verbose="ERROR")
    picks = mne.pick_types(raw.info, meg="mag")
    pos = np.array([raw.info["chs"][p]["loc"][:3] for p in picks]); x, z = pos[:, 0], pos[:, 2]
    tp = (np.abs(x) > np.percentile(np.abs(x), 70)) & (z < np.percentile(z, 45)); roi = np.where(tp)[0]
    for i, code in enumerate(ti):
        if int(code) not in CODEMAP:
            continue
        cname = CODEMAP[int(code)].replace(".MPG", "")
        (env250, dur), (env100, _) = ENV[CODEMAP[int(code)]]["250"], ENV[CODEMAP[int(code)]]["100"]
        onset = si[i, 0]/SF_MEG + ANCHOR_OFFSET
        s0 = int((onset-1)*SF_MEG); s1 = int((onset+dur+1)*SF_MEG)
        d = raw.get_data(picks=picks, start=max(0, s0), stop=min(s1, raw.n_times))
        # --- PSD (channel-averaged Welch), on raw clip data ---
        ff, P = welch(d, fs=SF_MEG, nperseg=int(4*SF_MEG), axis=1)
        sel = (ff >= 1) & (ff <= 45); P = P[:, sel].mean(0); ff = ff[sel]
        psd_freqs = ff
        bandp = {b: float(P[(ff >= lo) & (ff < hi)].mean()) for b, (lo, hi) in BANDS.items()}
        # --- envelope tracking (250 Hz, 1-8 Hz), small +/-1s search ---
        meg = resample_poly(d, FS, int(round(SF_MEG)), axis=1)
        mf = bp(meg, 1, 8, FS); mf = (mf-mf.mean(1, keepdims=True))/mf.std(1, keepdims=True)
        ef = bp(env250, 1, 8, FS); ef = (ef-ef.mean())/ef.std(); Ne = len(ef)
        ef_null = np.roll(ef, Ne//2)                       # circular-shift null (breaks true alignment)
        base = int(1*FS)  # onset position in the loaded segment
        def track(e):
            C = np.zeros((len(roi), int(2*FS)))
            for j, ch in enumerate(roi):
                c = correlate(mf[ch], e, mode="full")/Ne
                C[j] = np.interp(np.arange(base-int(1*FS), base+int(1*FS)), np.arange(c.size)-(Ne-1), c)
            return float(np.sqrt(np.mean(C**2, 0)).max())
        env_track_r = track(ef); env_track_r_null = track(ef_null)
        # --- reproduce 2.3.2: theta power-envelope (0.8Hz lowpass) vs acoustic env ---
        meg100 = resample_poly(d, 100, int(round(SF_MEG)), axis=1)
        th = bp(meg100, 4, 8, 100); pe = np.abs(hilbert(th, axis=1))
        b2, a2 = butter(4, 0.8/(100/2), "low"); pe = filtfilt(b2, a2, pe, axis=1)
        ae = filtfilt(b2, a2, env100)                     # acoustic env, same lowpass
        o100 = int(1*100); n = min(pe.shape[1]-o100, len(ae))
        peR = pe[roi, o100:o100+n]; aer = ae[:n]; aer = (aer-aer.mean())/aer.std()
        aer_null = np.roll(aer, len(aer)//2)
        def slowr(a):
            rr = [np.corrcoef(peR[k]-peR[k].mean(), a)[0, 1] for k in range(len(roi))]
            return float(np.nanmean(np.abs(rr)))
        slowpow_env_r = slowr(aer); slowpow_env_r_null = slowr(aer_null)
        S = store[cname]
        S["psd"].append(P); S["bandpow"].append([bandp[b] for b in BANDS])
        S["env_track_r"].append(env_track_r); S["slowpow_env_r"].append(slowpow_env_r)
        S["env_track_r_null"].append(env_track_r_null); S["slowpow_env_r_null"].append(slowpow_env_r_null)
        S["subj"].append(tag)
    print(f"{tag} done", flush=True)

np.savez(os.path.join(OUT, "meg_extract.npz"),
         psd_freqs=psd_freqs, bands=list(BANDS.keys()),
         **{f"{c}_{k}": np.array(store[c][k]) for c in store for k in ["psd", "bandpow", "env_track_r", "slowpow_env_r", "env_track_r_null", "slowpow_env_r_null"]},
         Charismatic_subj=store["Charismatic"]["subj"], NonCharismatic_subj=store["Non-Charismatic"]["subj"])
print("saved meg_extract.npz")
