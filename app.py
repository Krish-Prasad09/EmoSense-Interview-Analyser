from flask import Flask, request, render_template, jsonify
from emotion import predict_emotion

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB max upload

ALLOWED_EXTENSIONS = {"wav", "mp3", "ogg", "webm", "m4a", "flac"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]

    if audio_file.filename == "" and audio_file.content_length == 0:
        return jsonify({"error": "Empty file"}), 400

    try:
        audio_bytes = audio_file.read()
        if len(audio_bytes) == 0:
            return jsonify({"error": "Empty audio data"}), 400

        result = predict_emotion(audio_bytes)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyse_speech", methods=["POST"])
def analyse_speech_route():
    """
    Speech-to-text + communication quality analysis.
    Accepts same audio formats as /predict.
    Returns: transcript, highlighted_transcript, scores, tips, etc.
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]

    try:
        audio_bytes = audio_file.read()
        if len(audio_bytes) == 0:
            return jsonify({"error": "Empty audio data"}), 400

        from speech_analysis import analyse_speech
        result = analyse_speech(audio_bytes)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyse_full", methods=["POST"])
def analyse_full():
    """
    Combined endpoint: runs acoustic emotion analysis (Wav2Vec2) AND speech analysis
    (Whisper transcription + communication quality) + semantic text emotion analysis.
    Computes a multimodal late-fusion average so the dominant emotion shown on top
    is the balanced average of vocal tone and transcript sentiment.
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]

    try:
        audio_bytes = audio_file.read()
        if len(audio_bytes) == 0:
            return jsonify({"error": "Empty audio data"}), 400

        # ── 1. Acoustic Emotion analysis (Wav2Vec2) ────────────────────────
        audio_emotion_result = predict_emotion(audio_bytes)

        # ── 2. Speech analysis (Whisper + rule engine) ─────────────────────
        from speech_analysis import analyse_speech, predict_text_emotion
        speech_result = analyse_speech(audio_bytes)

        # ── 3. Semantic Emotion analysis from Transcript (DistilRoBERTa) ───
        transcript_text = speech_result.get("transcript", "")
        text_emotion_result = predict_text_emotion(transcript_text)

        # ── 4. Multimodal Late Fusion (Average of Audio + Text) ────────────
        audio_probs = audio_emotion_result.get("all_probs", {})
        text_probs = text_emotion_result.get("all_probs", {})

        word_count = len(transcript_text.strip().split())
        # If very short or empty transcript, rely primarily on vocal tone
        if word_count < 4:
            audio_weight, text_weight = 0.85, 0.15
        else:
            audio_weight, text_weight = 0.50, 0.50

        emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad"]
        fused_probs = {}
        for em in emotions:
            a_p = float(audio_probs.get(em, 0.0))
            t_p = float(text_probs.get(em, a_p))
            fused_probs[em] = round(audio_weight * a_p + text_weight * t_p, 1)

        fused_dominant = max(fused_probs, key=lambda k: fused_probs[k])
        fused_confidence = fused_probs[fused_dominant]

        from emotion import FEEDBACK
        feedback = FEEDBACK.get(fused_dominant, {
            "score": 50,
            "tone": fused_dominant.title(),
            "tip": "Keep practicing!",
            "color": "#64748b"
        })

        # Blended interview score
        text_score = feedback["score"]
        audio_score = audio_emotion_result.get("score", 50)
        fused_score = round(audio_weight * audio_score + text_weight * text_score)

        fused_emotion = {
            "emotion":    fused_dominant,
            "confidence": fused_confidence,
            "all_probs":  fused_probs,
            "score":      fused_score,
            "tone":       feedback["tone"],
            "tip":        feedback["tip"],
            "color":      feedback["color"],
            "duration":   audio_emotion_result.get("duration", 0),
            "timeline":   audio_emotion_result.get("timeline", []),
            "fusion_breakdown": {
                "audio_dominant":   audio_emotion_result.get("emotion"),
                "audio_confidence": audio_emotion_result.get("confidence"),
                "text_dominant":    text_emotion_result.get("dominant"),
                "text_confidence":  text_emotion_result.get("confidence"),
                "weights":          f"{int(audio_weight*100)}% Audio + {int(text_weight*100)}% Text"
            }
        }

        return jsonify({
            "emotion": fused_emotion,
            "speech":  speech_result,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print(" EmoSense by Group 17 starting at http://127.0.0.1:5000")
    app.run(debug=False, port=5000)
    # app.run(debug=False, host="0.0.0.0", port=7860)

    