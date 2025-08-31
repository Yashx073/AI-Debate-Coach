import os
import json
import streamlit as st
import matplotlib.pyplot as plt

from src.stt import transcribe
from src.nlp import summarize_stats
from src.voice import analyze_voice
from src.posture import analyze_posture
from src.feedback import generate_feedback
from src.youtube_search import search_videos, best_timestamps
from src.audio import extract_wav_from_video # This will be needed for your full pipeline


# --- Streamlit UI ---
st.set_page_config(page_title="AI Debate Coach", page_icon="🎤", layout="wide")
st.title("AI Debate Coach 🎤")

uploaded_file = st.file_uploader("Upload Debate Video (MP4, WAV, MP3)", type=["mp4", "wav", "mp3"])

if uploaded_file:
    # --- Processing pipeline ---
    with st.status("Processing video...", expanded=True) as status:
        # Save file temporarily
        input_path = os.path.join("temp", uploaded_file.name)
        os.makedirs("temp", exist_ok=True)
        with open(input_path, "wb") as f:
            f.write(uploaded_file.read())

        status.update(label="Extracting audio...")
        wav_path = extract_wav_from_video(uploaded_file.read())
        
        status.update(label="Transcribing (STT)...")
        text, duration = transcribe(wav_path)
        st.success("Transcription complete.")

        status.update(label="Analyzing transcript (NLP)...")
        nlp_stats = summarize_stats(text, duration)
        
        status.update(label="Analyzing voice...")
        voice_stats = analyze_voice(wav_path)

        status.update(label="Analyzing posture...")
        posture_stats = analyze_posture(input_path)

        status.update(label="Searching YouTube...")
        # Re-integrating the YouTube search functionality
        yt_items = search_videos(nlp_stats.get("keywords", [])[:5], max_results=5)
        for it in yt_items:
            ts = best_timestamps(it["video_id"], nlp_stats.get("keywords", []), top_k=2)
            it["timestamps"] = ts

        status.update(label="Generating feedback...")
        tips = generate_feedback(nlp_stats, voice_stats, posture_stats)
        status.update(label="Analysis complete!", state="complete")

    # --- Results Display ---
    st.subheader("Transcript")
    st.text_area("Transcript", text, height=200)

    st.subheader("NLP Analysis")
    col1, col2, col3 = st.columns(3)
    col1.metric("Words", nlp_stats["word_count"])
    col2.metric("WPM", nlp_stats["wpm"])
    col3.metric("Sentiment", nlp_stats["sentiment"])

    st.write("Filler Words:", dict(nlp_stats["fillers"]))
    st.write("Predicted Topic:", nlp_stats.get("topic"))
    st.write("Role Prediction:", nlp_stats.get("role"))

    # Filler word plot
    fig, ax = plt.subplots()
    fillers = nlp_stats["fillers"]
    if fillers:
        df = pd.DataFrame({"word": list(fillers.keys()), "count": list(fillers.values())})
        ax.bar(df["word"], df["count"])
        ax.set_title("Filler Word Distribution")
        ax.set_xlabel("Filler")
        ax.set_ylabel("Count")
    st.pyplot(fig)
    
    st.subheader("Voice Analysis")
    voice_stats = analyze_voice(input_path)
    pitch = voice_stats.get("pitch", {})
    st.metric("Pitch Mean (Hz)", f"{pitch.get('mean', 0):.1f}")
    st.metric("Pitch Std (Hz)", f"{pitch.get('std', 0):.1f}")
    st.metric("Pitch Min (Hz)", f"{pitch.get('min', 0):.1f}")
    st.metric("Pitch Max (Hz)", f"{pitch.get('max', 0):.1f}")

    st.subheader("Posture Analysis")
    st.write(posture_stats)

    st.subheader("YouTube Recommendations")
    if yt_items:
        for it in yt_items:
            url = it["url"]
            stamps = it.get("timestamps", [])
            if stamps:
                links = ", ".join([f"[{t}s]({url}&t={t}s)" for t in stamps])
                st.markdown(f"**{it['title']}** — {it['channel']} | {links}")
            else:
                st.markdown(f"**{it['title']}** — {it['channel']} | [{url}]({url})")
    else:
        st.warning("No YouTube API key configured or no results found.")

    st.subheader("Improvement Tips")
    tips = generate_feedback(nlp_stats, voice_stats, posture_stats)
    for t in tips:
        st.markdown(f"- {t}")

    # --- Download JSON Report ---
    st.download_button(
        "Download JSON Report",
        data=json.dumps({
            "transcript": text,
            "nlp": nlp_stats,
            "voice": voice_stats,
            "posture": posture_stats,
            "tips": tips,
        }, indent=2),
        file_name="debate_report.json",
        mime="application/json",
    )
    
    # Clean up temporary files
    os.remove(input_path)
    os.remove(wav_path)

else:
    st.info("Upload a video to get started.")