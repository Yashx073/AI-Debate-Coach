def generate_feedback(nlp_stats, voice_stats, posture_stats):
    """Generate human-readable improvement tips."""
    tips = []

    # Speed feedback
    wpm = nlp_stats.get("wpm", 0)
    if wpm < 110:
        tips.append("Try speaking faster for more energy.")
    elif wpm > 170:
        tips.append("Try slowing down for better clarity.")

    # Fillers
    if sum(nlp_stats.get("fillers", {}).values()) > 5:
        tips.append("Reduce filler words like 'um', 'uh', or 'like'.")

    # Sentiment
    sentiment = nlp_stats.get("sentiment")
    if sentiment == "negative":
        tips.append("Tone sounded negative; aim for more positive framing.")
    elif sentiment == "neutral":
        tips.append("Tone was neutral; add more emphasis for persuasion.")

    # Topic feedback
    topic = nlp_stats.get("topic")
    if topic != "N/A":
        tips.append(f"Main topic detected: {topic}. Stay focused to strengthen arguments.")

    # Party feedback
    party = nlp_stats.get("party")
    if party != "N/A":
        tips.append(f"Your language leaned towards {party}; balance it for broader appeal.")

    # Role feedback
    role = nlp_stats.get("role")
    if role != "N/A":
        tips.append(f"Predicted role: {role}. Adjust speaking style accordingly.")

    # Voice
    if voice_stats:
        pitch_std = voice_stats.get("pitch", {}).get("std", 0)
        if pitch_std < 20:
            tips.append("Pitch variation was low; add more vocal variety.")

    # Posture
    if posture_stats:
        if posture_stats.get("confidence", 0) < 0.5:
            tips.append("Body language seemed hesitant; keep an upright confident posture.")

    return tips
