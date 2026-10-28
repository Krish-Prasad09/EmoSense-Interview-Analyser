"""
Speech-to-text (Whisper) + Communication Quality Analysis:

Scores reported:
  • Filler words     — "um", "uh", "like", "you know", "basically", etc.
  • Grammar          — rule-based checks (subject-verb agreement, double negatives, etc.)
  • Sentence clarity — avg words-per-sentence, passive-voice ratio, long-sentence ratio
  • Vocabulary       — type-token ratio (lexical diversity)
  • Pace             — words per ACTIVE SPEAKING MINUTE (not total clip length).
                       Whisper timestamps are used to measure only the segments where
                       the candidate is actually speaking, so a 5-minute recording where
                       the interviewer talks for 3 minutes still gives a fair WPM reading.
  • Overall          — weighted composite

All of this runs 100 % locally (no API key needed).
Whisper model is downloaded once and cached by HuggingFace.
"""

import re
import math
import os
import io
import tempfile
import subprocess
import numpy as np

# ── Load Whisper at startup (same as emotion model) ───────────────────────

# loading it once at startup so we don't reload the model on every request
# whisper-small is good enough for interview audio and not too heavy
# return_timestamps=True is important — we need chunk timings later for WPM

from transformers import pipeline
print("Loading Whisper STT model...")
_whisper_pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    generate_kwargs={"language": "english"},
    return_timestamps=True,
)
print(" WHISPER loaded.")

def _get_whisper():
    return _whisper_pipe


# ── Text Emotion Model (DistilRoBERTa) ────────────────────────────────────
# Classifies emotion from Whisper's transcribed text.
# Maps 7 DistilRoBERTa emotions to the 6 CREMA-D target classes:
# anger -> angry, disgust -> disgust, fear -> fear, joy -> happy, neutral -> neutral, sadness -> sad
_text_emotion_pipe = None

TEXT_EMOTION_MAP = {
    "anger":   "angry",
    "disgust": "disgust",
    "fear":    "fear",
    "joy":     "happy",
    "neutral": "neutral",
    "sadness": "sad",
}

def _get_text_emotion_pipe():
    global _text_emotion_pipe
    if _text_emotion_pipe is None:
        print("Loading Text Emotion model (DistilRoBERTa)...")
        _text_emotion_pipe = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
        )
        print(" TEXT EMOTION model loaded.")
    return _text_emotion_pipe


def predict_text_emotion(transcript: str) -> dict:
    """
    Predict 6-class emotion distribution from transcribed text using DistilRoBERTa.
    Returns:
        dominant: str (e.g. 'happy')
        confidence: float
        all_probs: dict of {emotion: percentage}
    """
    all_emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad"]
    if not transcript or len(transcript.strip()) < 4:
        return {
            "dominant": "neutral",
            "confidence": 100.0,
            "all_probs": {e: (100.0 if e == "neutral" else 0.0) for e in all_emotions}
        }

    try:
        pipe = _get_text_emotion_pipe()
        raw_outputs = pipe(transcript[:512])[0]

        scores = {e: 0.0 for e in all_emotions}
        for item in raw_outputs:
            mapped = TEXT_EMOTION_MAP.get(item["label"].lower())
            if mapped in scores:
                scores[mapped] = float(item["score"])

        total = sum(scores.values())
        if total > 0:
            all_probs = {e: round((s / total) * 100, 1) for e, s in scores.items()}
        else:
            all_probs = {e: round(100.0 / len(all_emotions), 1) for e in all_emotions}

        dominant = max(all_probs, key=lambda k: all_probs[k])
        confidence = all_probs[dominant]

        return {
            "dominant": dominant,
            "confidence": confidence,
            "all_probs": all_probs
        }
    except Exception as e:
        print(f"Warning: Text emotion prediction fallback: {e}")
        return {
            "dominant": "neutral",
            "confidence": 100.0,
            "all_probs": {e: (100.0 if e == "neutral" else 0.0) for e in all_emotions}
        }


def _detect_suffix(audio_bytes: bytes) -> str:
    # audio formats have fixed "magic bytes" at the start, checking those
    # to figure out the file type without relying on a filename/extension
    if audio_bytes[:4] == b"RIFF":               return ".wav"
    if audio_bytes[:3] == b"ID3":                return ".mp3"
    if audio_bytes[:4] == b"fLaC":               return ".flac"
    if audio_bytes[:4] == b"OggS":               return ".ogg"
    if audio_bytes[4:8] == b"ftyp":              return ".m4a"   # ftyp is at offset 4 for mp4/m4a
    if audio_bytes[:4] == b"\x1a\x45\xdf\xa3":  return ".webm"
    return ".webm"   # browser recordings are almost always webm so safe default


def _to_wav_numpy(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """
    Convert any audio format → numpy float32 array at 16 kHz mono.
    Returns (array, sample_rate).
    """
    import torchaudio, torch

    # try torchaudio first since it's faster than spawning ffmpeg
    try:
        wf, sr = torchaudio.load(io.BytesIO(audio_bytes))
    except Exception:
        # torchaudio can't handle this codec, fall back to ffmpeg
        # write to a temp file, convert, then read back
        suffix  = _detect_suffix(audio_bytes)
        tmp_in  = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp_out = tempfile.NamedTemporaryFile(suffix=".wav",  delete=False)
        tmp_in.close(); tmp_out.close()
        try:
            with open(tmp_in.name, "wb") as f: f.write(audio_bytes)
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_in.name,
                 "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", tmp_out.name],
                capture_output=True, timeout=120,
            )
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg: {result.stderr.decode()}")
            wf, sr = torchaudio.load(tmp_out.name)
        finally:
            # cleanup temp files whether or not something went wrong
            for p in (tmp_in.name, tmp_out.name):
                if os.path.exists(p): os.remove(p)

    # whisper only works at 16kHz
    if sr != 16000:
        wf = torchaudio.functional.resample(wf, sr, 16000)
        sr = 16000

    # whisper expects mono, so average left+right if stereo
    if wf.shape[0] > 1:
        wf = wf.mean(dim=0, keepdim=True)

    return wf.squeeze().numpy().astype(np.float32), sr


# ── Transcription ──────────────────────────────────────────────────────────

def transcribe(audio_bytes: bytes) -> dict:
    """
    Run Whisper on audio_bytes.

    Returns:
        text               – full transcript string
        chunks             – raw Whisper chunk list (each has 'timestamp')
        duration_seconds   – total length of the recording
        speaking_seconds   – sum of durations of speech segments only.
                             This is the denominator used for WPM so that
                             silence / interviewer questions don't make the
                             candidate look like they were speaking slowly.
    """
    audio_np, sr = _to_wav_numpy(audio_bytes)
    duration_sec = len(audio_np) / sr

    pipe = _get_whisper()

    # 30s chunks with 5s overlap so words at chunk boundaries don't get dropped
    result = pipe(
        {"array": audio_np, "sampling_rate": sr},
        chunk_length_s=30,
        stride_length_s=5,
    )

    text   = result.get("text", "").strip()
    chunks = result.get("chunks", [])

    # ── Measure actual speaking time from per-chunk timestamps ────────────
    # each chunk has a timestamp like [start, end] in seconds
    # adding those up gives only the time the person was actually talking,
    # not the full recording length (which might include interviewer speaking, silence, etc.)
    speaking_seconds = 0.0
    for chunk in chunks:
        ts = chunk.get("timestamp")
        if ts and len(ts) == 2 and ts[0] is not None and ts[1] is not None:
            chunk_dur = float(ts[1]) - float(ts[0])
            if chunk_dur > 0:
                speaking_seconds += chunk_dur

    # if whisper didn't return timestamps for some reason, just use total duration
    if speaking_seconds < 1.0:
        speaking_seconds = duration_sec

    return {
        "text":             text,
        "chunks":           chunks,
        "duration_seconds": round(duration_sec, 1),
        "speaking_seconds": round(speaking_seconds, 1),
    }


# ── Communication analysis ─────────────────────────────────────────────────

# covers both short fillers (um, uh) and longer phrases (you know what i mean)
FILLERS = {
    "um", "uh", "er", "ah", "hmm", "umm", "uhh", "err",
    "like", "basically", "literally", "honestly", "actually",
    "you know", "you know what i mean", "i mean",
    "sort of", "kind of", "kinda", "sorta",
    "right", "okay so", "so yeah", "and stuff",
    "at the end of the day", "to be honest", "to be fair",
    "well", "anyway", "whatever",
}

# multi-word ones need to be checked first (longest first) so we don't accidentally
# match "you" before we get a chance to match "you know"
MULTI_FILLERS = sorted(
    [f for f in FILLERS if " " in f],
    key=lambda x: -len(x)
)

SINGLE_FILLERS = {f for f in FILLERS if " " not in f}

# vague words that weaken the answer without adding meaning
WEAK_WORDS = {
    "thing", "stuff", "things", "lot", "lots", "very", "really",
    "quite", "pretty", "just", "maybe", "perhaps", "probably",
    "might", "could", "somewhat", "a bit",
}

# regex pattern → feedback message
# these catch the most common spoken grammar mistakes
GRAMMAR_ISSUES = [
    (r"\bi done\b",            "Incorrect: 'I done' → use 'I did' or 'I have done'"),
    (r"\bthey was\b",          "Subject-verb disagreement: 'they was' → 'they were'"),
    (r"\bwe was\b",            "Subject-verb disagreement: 'we was' → 'we were'"),
    (r"\byou was\b",           "Subject-verb disagreement: 'you was' → 'you were'"),
    (r"\bi seen\b",            "Incorrect: 'I seen' → 'I saw' or 'I have seen'"),
    (r"\bi have went\b",       "Incorrect: 'I have went' → 'I have gone'"),
    (r"\bshould of\b",         "Common error: 'should of' → 'should have'"),
    (r"\bwould of\b",          "Common error: 'would of' → 'would have'"),
    (r"\bcould of\b",          "Common error: 'could of' → 'could have'"),
    (r"\bmore better\b",       "Double comparative: 'more better' → 'better'"),
    (r"\bmost best\b",         "Double superlative: 'most best' → 'best'"),
    (r"\bdon't have no\b",     "Double negative detected"),
    (r"\bcan't do nothing\b",  "Double negative detected"),
    (r"\bain't\b",             "Informal: 'ain't' — avoid in interviews"),
    (r"\bgonna\b",             "Informal: 'gonna' → 'going to'"),
    (r"\bwanna\b",             "Informal: 'wanna' → 'want to'"),
    (r"\bgotta\b",             "Informal: 'gotta' → 'have to' / 'got to'"),
    (r"\bdunno\b",             "Informal: 'dunno' → 'I don't know'"),
    (r"\byeah\b",              "Informal: 'yeah' → 'yes'"),
    (r"\bnope\b",              "Informal: 'nope' → 'no'"),
]

# matches constructions like "was done", "is being reviewed", "were told"
PASSIVE_PATTERN = re.compile(
    r"\b(was|were|is|are|been|be|being)\s+(being\s+)?\w+ed\b", re.IGNORECASE
)


def _find_fillers(text_lower: str) -> list[str]:
    found = []
    temp  = text_lower

    # scan multi-word fillers first, then blank out that region with spaces
    # so the single-word pass doesn't double count words already in a phrase
    for mf in MULTI_FILLERS:
        count = temp.count(mf)
        for _ in range(count):
            found.append(mf)
        temp = temp.replace(mf, " " * len(mf))

    # now pick up single-word fillers from whatever's left
    words = re.findall(r"\b\w+\b", temp)
    for w in words:
        if w in SINGLE_FILLERS:
            found.append(w)

    return found


def _sentences(text: str) -> list[str]:
    # split on . ! ? — not perfect but works fine for speech transcripts
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def analyse_communication(transcript_text: str, speaking_seconds: float) -> dict:
    """
    Full communication quality analysis.

    speaking_seconds should be the ACTIVE SPEAKING TIME derived from
    Whisper timestamps — not the total clip duration. This ensures WPM
    reflects the candidate's delivery speed only, unaffected by the
    interviewer asking questions or natural pauses between turns.

    Returns a rich dict with scores and detailed feedback.
    """
    if not transcript_text or len(transcript_text.strip()) < 10:
        return {"error": "Transcript too short to analyse."}

    text       = transcript_text.strip()
    text_lower = text.lower()
    words      = re.findall(r"\b\w+\b", text_lower)
    sentences  = _sentences(text)

    word_count     = len(words)
    sentence_count = max(len(sentences), 1)  # avoid div by zero

    if word_count < 5:
        return {"error": "Too few words to analyse (< 5)."}

    # ── 1. Filler words ───────────────────────────────────────────────────
    filler_instances = _find_fillers(text_lower)
    filler_count     = len(filler_instances)
    filler_rate      = filler_count / word_count

    # 0 fillers = 100, hits 0 once fillers are 20% of words
    filler_score = max(0, round(100 - (filler_rate / 0.20) * 100))
    filler_freq  = {}
    for f in filler_instances:
        filler_freq[f] = filler_freq.get(f, 0) + 1
    top_fillers = sorted(filler_freq.items(), key=lambda x: -x[1])[:5]

    # ── 2. Grammar ────────────────────────────────────────────────────────
    grammar_issues = []
    for pattern, msg in GRAMMAR_ISSUES:
        if re.search(pattern, text_lower):
            grammar_issues.append(msg)

    # -10 per issue, so 10 issues = 0
    grammar_score = max(0, 100 - len(grammar_issues) * 10)

    # ── 3. Sentence clarity ───────────────────────────────────────────────
    avg_words_per_sent = word_count / sentence_count
    sentence_lengths   = [len(re.findall(r"\b\w+\b", s)) for s in sentences]

    # anything over 35 words is hard to follow when spoken out loud
    long_sent_count = sum(1 for l in sentence_lengths if l > 35)
    long_sent_ratio = long_sent_count / sentence_count

    passive_matches = PASSIVE_PATTERN.findall(text)
    passive_ratio   = len(passive_matches) / sentence_count

    weak_count = sum(1 for w in words if w in WEAK_WORDS)
    weak_ratio = weak_count / word_count

    # start from 100 and deduct for each problem
    clarity_score = 100
    clarity_score -= long_sent_ratio  * 30
    clarity_score -= passive_ratio    * 20
    clarity_score -= weak_ratio       * 30
    # ideal spoken sentence is around 12-20 words
    if avg_words_per_sent > 25:
        clarity_score -= (avg_words_per_sent - 25) * 1.5
    elif avg_words_per_sent < 6:
        clarity_score -= (6 - avg_words_per_sent) * 3
    clarity_score = max(0, min(100, round(clarity_score)))

    # ── 4. Vocabulary diversity (type-token ratio) ────────────────────────
    unique_words  = set(words)
    ttr           = len(unique_words) / word_count

    # raw TTR is biased — longer answers repeat words more by nature
    # so we look at only the first 50 words to keep it fair across lengths
    window = 50
    if word_count >= window:
        cttr = len(set(words[:window])) / window
    else:
        cttr = ttr

    vocab_score = min(100, round(cttr * 150))  # ~67% unique in 50 words → score of 100

    # ── 5. Pace — based on active speaking time only ──────────────────────
    # using speaking_seconds here, not total clip duration, so the interviewer
    # talking for 3 minutes doesn't make the candidate look slow
    wpm = (word_count / speaking_seconds) * 60 if speaking_seconds > 0 else 0

    IDEAL_LOW, IDEAL_HIGH = 130, 160  # good interview pace range
    if IDEAL_LOW <= wpm <= IDEAL_HIGH:
        pace_score = 100
    elif wpm < IDEAL_LOW:
        pace_score = max(0, round(100 - (IDEAL_LOW - wpm) * 1.2))
    else:
        # steeper penalty for going too fast vs too slow
        pace_score = max(0, round(100 - (wpm - IDEAL_HIGH) * 1.5))

    if wpm < 80:
        pace_feedback = "Very slow — try to be more fluent and natural."
    elif wpm < 130:
        pace_feedback = "Slightly slow — aim for a more conversational pace."
    elif wpm <= 160:
        pace_feedback = "Great pace — clear and easy to follow."
    elif wpm <= 200:
        pace_feedback = "A bit fast — slow down so the interviewer can follow."
    else:
        pace_feedback = "Too fast — significantly slow down your delivery."

    # ── 6. Composite score ────────────────────────────────────────────────
    # filler + grammar + clarity each get 0.25 since those are most noticeable
    # pace only gets 0.10 since being slightly off pace isn't a dealbreaker
    overall_score = round(
        filler_score  * 0.25 +
        grammar_score * 0.25 +
        clarity_score * 0.25 +
        vocab_score   * 0.15 +
        pace_score    * 0.10
    )

    # ── 7. Summary tips ───────────────────────────────────────────────────
    tips = []
    if filler_score < 70:
        worst = top_fillers[0][0] if top_fillers else "filler words"
        tips.append(f"Reduce filler words — you used '{worst}' {top_fillers[0][1] if top_fillers else ''} times. Pause silently instead.")
    if grammar_score < 80 and grammar_issues:
        tips.append(f"Grammar: {grammar_issues[0]}")
    if clarity_score < 70:
        if long_sent_ratio > 0.2:
            tips.append("Break long sentences into shorter ones — aim for under 20 words per sentence.")
        if passive_ratio > 0.3:
            tips.append("Use active voice more: say 'I led the team' not 'The team was led by me'.")
    if vocab_score < 60:
        tips.append("Vary your vocabulary — avoid repeating the same words. Use specific, concrete language.")
    if pace_score < 70:
        tips.append(pace_feedback)

    if not tips:
        tips.append("Excellent communication! Keep up the confident, clear delivery.")

    # wrap filler words in <mark> tags so the frontend can highlight them
    # sorting longest first to avoid partial replacements inside longer phrases
    highlighted = text
    all_filler_words = sorted(set(filler_instances), key=lambda x: -len(x))
    for f in all_filler_words:
        pattern = re.compile(r'\b' + re.escape(f) + r'\b', re.IGNORECASE)
        highlighted = pattern.sub(f'<mark class="filler">{f}</mark>', highlighted)

    return {
        "overall_score":   overall_score,
        "filler_score":    filler_score,
        "grammar_score":   grammar_score,
        "clarity_score":   clarity_score,
        "vocab_score":     vocab_score,
        "pace_score":      pace_score,

        "word_count":          word_count,
        "sentence_count":      sentence_count,
        "avg_words_per_sent":  round(avg_words_per_sent, 1),
        "filler_count":        filler_count,
        "filler_rate_pct":     round(filler_rate * 100, 1),
        "top_fillers":         [{"word": w, "count": c} for w, c in top_fillers],
        "grammar_issues":      grammar_issues,
        "passive_ratio_pct":   round(passive_ratio * 100, 1),
        "wpm":                 round(wpm, 1),
        "speaking_seconds":    round(speaking_seconds, 1),
        "pace_feedback":       pace_feedback,
        "tips":                tips,

        "highlighted_transcript": highlighted,
    }


# ── Combined entry point ───────────────────────────────────────────────────

def analyse_speech(audio_bytes: bytes) -> dict:
    """
    Transcribe audio and run full communication analysis.
    Returns combined dict safe to jsonify().
    """
    stt          = transcribe(audio_bytes)
    text         = stt["text"]
    duration_sec = stt["duration_seconds"]
    speaking_sec = stt["speaking_seconds"]

    # pass speaking_sec (not duration) so WPM isn't skewed by silence/interviewer
    comm = analyse_communication(text, speaking_sec)

    return {
        "transcript":       text,
        "duration_seconds": duration_sec,
        "speaking_seconds": speaking_sec,
        "communication":    comm,
    }
