"""Study 1 (charisma): Hebrew ASR + CTC forced alignment of the two clips -> word onsets.

Extracts 16 kHz mono audio from each MPG, writes:
  reanalysis/asr/<tag>.wav                (tag in {Charismatic, Non_Charismatic})
  reanalysis/asr/<tag>_whisper.json
  reanalysis/asr/<tag>_words.json  : [{word, norm, start, end, whisper_start, prob, score}]

Usage: python align_speech.py
"""
import json, os, re, sys, subprocess
import numpy as np, soundfile as sf, torch
from torchaudio.functional import forced_align
from transformers import Wav2Vec2ForCTC
from huggingface_hub import hf_hub_download
from faster_whisper import WhisperModel
import imageio_ffmpeg

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"G:/Barak1"; ASR = os.path.join(ROOT, "reanalysis", "asr"); os.makedirs(ASR, exist_ok=True)
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
CLIP = {"Charismatic": "Charismatic.MPG", "Non_Charismatic": "Non-Charismatic.MPG"}
SR = 16000
CTC = "imvladikon/wav2vec2-xls-r-300m-hebrew"
HEB = re.compile(r"[^א-ת]")

def norm(w): return HEB.sub("", w)

# ---- audio extraction ----
def extract(tag):
    wav = os.path.join(ASR, f"{tag}.wav")
    if not os.path.exists(wav):
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", os.path.join(ROOT, CLIP[tag]),
                        "-vn", "-ac", "1", "-ar", str(SR), "-f", "wav", wav], check=True)
    return wav

# ---- Whisper ----
def transcribe(tag, wav):
    out = os.path.join(ASR, f"{tag}_whisper.json")
    if os.path.exists(out):
        print(f"{tag}: whisper exists, skip", flush=True); return out
    model = WhisperModel("large-v3", device="cpu", compute_type="int8")
    segs, info = model.transcribe(wav, language="he", word_timestamps=True, beam_size=5,
                                  vad_filter=False, condition_on_previous_text=True)
    data, nw = [], 0
    for s in segs:
        ws = [{"word": w.word, "start": w.start, "end": w.end, "prob": w.probability} for w in (s.words or [])]
        nw += len(ws); data.append({"id": s.id, "start": s.start, "end": s.end, "text": s.text, "words": ws})
        print(f"  [{s.start:7.2f}-{s.end:7.2f}] {s.text.strip()[:70]}", flush=True)
    json.dump({"tag": tag, "wav": wav, "model": "large-v3", "duration": info.duration, "segments": data},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{tag}: {len(data)} segments, {nw} words", flush=True); return out

# ---- CTC forced alignment ----
_ctc = None; _vocab = None; _BLANK = None; _SEP = None
def ctc_load():
    global _ctc, _vocab, _BLANK, _SEP
    if _ctc is None:
        _ctc = Wav2Vec2ForCTC.from_pretrained(CTC).eval()
        _vocab = json.load(open(hf_hub_download(CTC, "vocab.json"), encoding="utf-8"))
        _BLANK = _vocab["[PAD]"]; _SEP = _vocab["|"]
        print(f"CTC vocab {len(_vocab)} | blank={_BLANK} sep={_SEP}", flush=True)

@torch.inference_mode()
def emissions(wav):
    x = torch.from_numpy(wav).float()[None]; x = (x - x.mean()) / (x.std() + 1e-9)
    return torch.log_softmax(_ctc(x).logits, dim=-1), x.shape[1]

def align_segment(wav, words):
    toks, owner = [], []
    for k, (_, nw) in enumerate(words):
        if toks: toks.append(_SEP); owner.append(-1)
        for ch in nw:
            i = _vocab.get(ch)
            if i is None: return {}
            toks.append(i); owner.append(k)
    if not toks: return {}
    logp, nsamp = emissions(wav); F = logp.shape[1]
    if F < len(toks) + 1: return {}
    path, scores = forced_align(logp, torch.tensor([toks], dtype=torch.int32), blank=_BLANK)
    path, scores = path[0].numpy(), scores[0].exp().numpy()
    sec = nsamp / SR / F
    spans = {}; ti = -1; prev = _BLANK
    for f, tok in enumerate(path):
        if tok == _BLANK: prev = tok; continue
        if tok != prev: ti += 1
        prev = tok
        if ti >= len(owner): break
        k = owner[ti]
        if k < 0: continue
        if k not in spans: spans[k] = [f, f, []]
        spans[k][1] = f; spans[k][2].append(scores[f])
    out = {}
    for k, (f0, f1, sc) in spans.items():
        out[words[k][0]] = (f0 * sec, (f1 + 1) * sec, float(np.mean(sc)))
    return out

def align(tag, wav_path):
    out = os.path.join(ASR, f"{tag}_words.json")
    if os.path.exists(out): print(f"{tag}: words exist, skip", flush=True); return
    ctc_load()
    js = json.load(open(os.path.join(ASR, f"{tag}_whisper.json"), encoding="utf-8"))
    wav, sr = sf.read(wav_path); assert sr == SR, sr
    if wav.ndim > 1: wav = wav.mean(1)
    words, dropped, unaligned = [], 0, 0
    for seg in js["segments"]:
        ws = seg["words"]; cand = [(i, norm(w["word"])) for i, w in enumerate(ws)]
        keep = [(i, nw) for i, nw in cand if nw]; dropped += len(cand) - len(keep)
        if not keep: continue
        t0 = max(0.0, seg["start"] - 0.20); t1 = min(len(wav) / SR, seg["end"] + 0.20)
        sl = wav[int(t0 * SR):int(t1 * SR)]
        try: sp = align_segment(sl, keep)
        except Exception as e: print(f"  seg {seg['id']}: align failed ({e})", flush=True); sp = {}
        for i, nw in keep:
            w = ws[i]
            if i in sp:
                a, b, s = sp[i]
                words.append(dict(word=w["word"].strip(), norm=nw, start=t0+a, end=t0+b,
                                  whisper_start=w["start"], prob=w["prob"], score=s, seg=seg["id"]))
            else:
                unaligned += 1
                words.append(dict(word=w["word"].strip(), norm=nw, start=w["start"], end=w["end"],
                                  whisper_start=w["start"], prob=w["prob"], score=float("nan"), seg=seg["id"]))
    words.sort(key=lambda d: d["start"])
    d = np.array([w["start"] - w["whisper_start"] for w in words if np.isfinite(w["score"])])
    json.dump({"tag": tag, "ctc": CTC, "n_words": len(words), "n_dropped": dropped,
               "n_unaligned": unaligned, "words": words}, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"{tag}: {len(words)} words ({dropped} non-Hebrew dropped, {unaligned} fell back); "
          f"CTC-vs-Whisper median {np.median(d):+.3f}s, |d|>0.1s in {(np.abs(d)>0.1).mean()*100:.0f}%", flush=True)

for tag in ["Charismatic", "Non_Charismatic"]:
    print(f"=== {tag} ===", flush=True)
    wav = extract(tag); transcribe(tag, wav); align(tag, wav)
print("S1 ASR+align done")
