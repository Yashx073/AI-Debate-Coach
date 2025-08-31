# AI Debate Coach – Streamlit MVP+


A Streamlit app that analyzes a user-uploaded debate video to provide transcript, delivery stats (WPM, fillers, sentiment), posture & voice insights, and YouTube recommendations with (optional) timestamps.


## Features
- Video upload → Audio extraction (FFmpeg)
- Speech-to-text (Whisper by default; optional Google STT)
- NLP stats: keywords (KeyBERT/YAKE), filler counts, WPM, sentiment (Transformers/TextBlob)
- Voice: pitch, pace, intonation (librosa)
- Posture: facing camera %, head tilt via MediaPipe
- YouTube search (YouTube Data API) + caption-based timestamp hints (if available)
- Feedback summary and improvement tips


## Quickstart
```bash
# 1) System deps: FFmpeg (required)
# macOS (brew): brew install ffmpeg
# Ubuntu/Debian: sudo apt-get update && sudo apt-get install -y ffmpeg


# 2) Create env & install
python -m venv .venv && source .venv/bin/activate # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt


# 3) Configure
cp .env.sample .env
# Add your YOUTUBE_API_KEY in .env


# 4) Run
streamlit run app.py
```


## Notes
- Whisper small/base models may require time and CPU/GPU. For faster results, use Google STT.
- All uploads are processed in-memory by default; adjust to S3/GCS for production.
- This is an MVP: tune thresholds/UX for your use-case.