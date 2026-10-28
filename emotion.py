import torch
import torchaudio
import numpy as np
import io
import subprocess
import tempfile
import os
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

# ── Load model once at startup ─────────────────────────────────────────────
MODEL_PATH = "./ser_wav2vec2_best"
# MODEL_PATH = "Krizzh/emosense-wav2vec2-crema-d"

SAMPLE_RATE    = 16_000
WINDOW_SECONDS = 4          # model was trained on 4-second windows — never change this
STEP_SECONDS   = 10         # emit one data-point every 10 s of interview audio
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SECONDS   # 64 000  (model input size)
STEP_SAMPLES   = SAMPLE_RATE * STEP_SECONDS     # 160 000 (hop between windows) how far to jump forward before taking the next 4-second snapshot, to build an emotion-over-time timeline for a full interview recording

print("Loading emotion model...")
processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
model     = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

id2label     = model.config.id2label
label2id     = model.config.label2id
EMOTION_LIST = [id2label[i] for i in range(len(id2label))]
print(f" MODEL LOADED SUCCESSFULLY. Emotions: {EMOTION_LIST}")


# ── Interview feedback map ─────────────────────────────────────────────────
FEEDBACK = {
    "angry": {
        "score": 30,
        "tone":  "⚠️ Aggressive",
        "tip":   "You sound frustrated or aggressive. Take a breath before answering — interviewers value composure under pressure.",
        "color": "#ef4444"
    },
    "disgust": {
        "score": 35,
        "tone":  "⚠️ Negative",
        "tip":   "Your tone may come across as dismissive. Try to reframe your language to sound solution-oriented and positive.",
        "color": "#f97316"
    },
    "fear": {
        "score": 45,
        "tone":  "😰 Nervous",
        "tip":   "Nervousness is very natural! Practice mock answers aloud daily. Slowing down your speech will help you sound more confident.",
        "color": "#a855f7"
    },
    "happy": {
        "score": 90,
        "tone":  "😊 Enthusiastic",
        "tip":   "Great energy! You sound positive and engaged — exactly what interviewers love to see. Keep it up!",
        "color": "#22c55e"
    },
    "neutral": {
        "score": 75,
        "tone":  "😐 Composed",
        "tip":   "You sound calm and collected. Try adding a bit more enthusiasm to stand out — show genuine excitement for the role.",
        "color": "#3b82f6"
    },
    "sad": {
        "score": 40,
        "tone":  "😔 Disengaged",
        "tip":   "You sound low-energy or disengaged. Sit upright, smile while speaking — it physically lifts your vocal tone.",
        "color": "#6366f1"
    },
}


def _detect_suffix(audio_bytes: bytes) -> str:
    """Guess file extension from magic bytes so ffmpeg picks the right demuxer."""
    if audio_bytes[:4] == b"RIFF":              return ".wav"
    if audio_bytes[:3] == b"ID3":               return ".mp3"
    if audio_bytes[:4] == b"fLaC":              return ".flac"
    if audio_bytes[:4] == b"OggS":              return ".ogg"
    if audio_bytes[4:8] == b"ftyp":             return ".m4a"
    if audio_bytes[:4] == b"\x1a\x45\xdf\xa3": return ".webm"
    return ".webm"


def _load_audio_bytes(audio_bytes: bytes):
    """Load audio bytes → (waveform tensor [1, N], sample_rate)."""
    # ── Attempt 1: torchaudio direct ──────────────────────────────────────
    try:
        waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))
        return waveform, sr
    except Exception:
        pass

    # ── Attempt 2: ffmpeg → WAV → torchaudio ──────────────────────────────
    suffix  = _detect_suffix(audio_bytes)
    tmp_in  = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_out = tempfile.NamedTemporaryFile(suffix=".wav",  delete=False)
    tmp_in.close(); tmp_out.close()

    try:
        with open(tmp_in.name, "wb") as f:
            f.write(audio_bytes) 
            # user uploaded speech.mp3 is stored in memory as audio_bytes and this writes audio_bytes into /tmp/tmpx8123.mp3... FFmpeg has actual MP3 file to read now

        result = subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_in.name,
             "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", tmp_out.name],
            capture_output=True, timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg error:\n{result.stderr.decode()}")

        waveform, sr = torchaudio.load(tmp_out.name)
        return waveform, sr

    finally:
        for p in (tmp_in.name, tmp_out.name):
            if os.path.exists(p):
                os.remove(p)


def _infer_window(speech_np: np.ndarray) -> tuple[str, float, dict]:
    """
    Run model on a single numpy array of shape (WINDOW_SAMPLES,).
    Returns (emotion_label, confidence_0_to_1, {label: pct}).
    """
    # Pad if shorter than window (only the last window can be short)
    if len(speech_np) < WINDOW_SAMPLES:
        speech_np = np.pad(speech_np, (0, WINDOW_SAMPLES - len(speech_np)))

    # Normalize
    max_val = np.max(np.abs(speech_np))
    if max_val > 1e-8:
        speech_np = speech_np / max_val

    inputs = processor(
        speech_np,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
        max_length=WINDOW_SAMPLES,
        truncation=True,
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    probs      = torch.softmax(logits, dim=-1)[0]
    pred_id    = probs.argmax().item()
    emotion    = id2label[pred_id]
    confidence = float(probs[pred_id])
    all_probs  = {id2label[i]: round(float(p) * 100, 1) for i, p in enumerate(probs)}

    return emotion, confidence, all_probs


# ── Core inference function ────────────────────────────────────────────────
def predict_emotion(audio_bytes: bytes) -> dict:
    """
    Accepts WAV / MP3 / OGG / FLAC / WebM.
    For audio longer than STEP_SECONDS the result includes a per-segment
    timeline so the frontend can draw an emotion-over-time chart.

    Returns dict with keys:
      emotion, confidence, all_probs, score, tone, tip, color   ← overall (dominant)
      timeline  ← list of {t_start, t_end, emotion, confidence, all_probs}
                   one entry per STEP_SECONDS of audio
    """
    waveform, sr = _load_audio_bytes(audio_bytes)

    # Resample to 16 kHz
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

    # Stereo → mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    speech     = waveform.squeeze().numpy()          # shape: (N,)
    total_samp = len(speech)
    total_sec  = total_samp / SAMPLE_RATE

    # ── Build segments ─────────────────────────────────────────────────────
    # Each segment is centered on a STEP_SECONDS hop but uses a 4-second
    # window that starts in the middle of the hop so the model sees real speech.
    #
    # For audio < STEP_SECONDS (e.g. 6-second clip) we produce a single segment.

    segments = []
    step = STEP_SAMPLES

    start = 0
    while start < total_samp:
        end       = min(start + step, total_samp)
        # The 4-second model window is taken from the start of each segment.
        win_end   = min(start + WINDOW_SAMPLES, total_samp)
        chunk     = speech[start:win_end].copy()

        t_start = start / SAMPLE_RATE
        t_end   = end   / SAMPLE_RATE

        emotion, confidence, all_probs = _infer_window(chunk)

        segments.append({
            "t_start":    round(t_start, 1),
            "t_end":      round(t_end, 1),
            "label":      f"{int(t_start//60):02d}:{int(t_start%60):02d}",   # "MM:SS"
            "emotion":    emotion,
            "confidence": round(confidence * 100, 1),
            "all_probs":  all_probs,
        })

        start += step

    # ── Overall / dominant emotion ─────────────────────────────────────────
    # Weighted by segment duration (last segment may be shorter).
    emotion_scores: dict[str, float] = {e: 0.0 for e in EMOTION_LIST}
    total_weight = 0.0

    for seg in segments:
        dur = seg["t_end"] - seg["t_start"]
        for e, pct in seg["all_probs"].items():
            emotion_scores[e] += pct * dur
        total_weight += dur

    if total_weight > 0:
        for e in emotion_scores:
            emotion_scores[e] /= total_weight

    dominant = max(emotion_scores, key=lambda e: emotion_scores[e])

    # Overall confidence = weighted average probability of dominant emotion
    overall_conf = round(emotion_scores[dominant], 1)
    overall_all  = {e: round(v, 1) for e, v in emotion_scores.items()}

    # ── Average interview score ────────────────────────────────────────────
    avg_score = round(
        sum(FEEDBACK.get(s["emotion"], {"score": 50})["score"] for s in segments) / len(segments)
    )

    feedback = FEEDBACK.get(dominant, {
        "score": 50, "tone": dominant.title(),
        "tip": "Keep practicing!", "color": "#64748b"
    })

    return {
        # ── overall summary ──
        "emotion":    dominant,
        "confidence": overall_conf,
        "all_probs":  overall_all,
        "score":      avg_score,
        "tone":       feedback["tone"],
        "tip":        feedback["tip"],
        "color":      feedback["color"],
        "duration":   round(total_sec, 1),
        # ── timeline ──
        "timeline":   segments,      # list of per-10-s dicts
    }
