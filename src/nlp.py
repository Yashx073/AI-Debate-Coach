import os
import re
import json
import joblib
from collections import Counter
from textblob import TextBlob

# Load trained models (replace with your actual paths)
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

try:
    topic_model = joblib.load(os.path.join(MODELS_DIR, "models\topic_model.pkl"))
except:
    topic_model = None

try:
    role_model = joblib.load(os.path.join(MODELS_DIR, "role_classifier.pkl"))
except:
    role_model = None


def detect_fillers(text: str):
    """Count filler words like 'um', 'uh', 'like'."""
    fillers = re.findall(r"\b(um+|uh+|like|you know|so)\b", text.lower())
    return Counter(fillers)


def sentiment_analysis(text: str):
    """Simple sentiment analysis using TextBlob."""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return "positive"
    elif polarity < -0.1:
        return "negative"
    else:
        return "neutral"


def predict_topic(text: str):
    """Predict topic using trained model."""
    if topic_model:
        return topic_model.predict([text])[0]
    return "N/A"


def predict_role(text: str, num_questions: int = 0):
    """Predict role (Moderator/Candidate)."""
    if role_model:
        # Example: concatenate text + structured feature
        features = [text, num_questions]
        return role_model.predict([features])[0]
    return "N/A"


def summarize_stats(text: str, duration: float, num_questions: int = 0):
    """Run all NLP stats and model predictions."""
    word_count = len(text.split())
    wpm = word_count / (duration / 60.0) if duration else 0
    fillers = detect_fillers(text)
    sentiment = sentiment_analysis(text)

    return {
        "word_count": word_count,
        "wpm": round(wpm, 2),
        "fillers": fillers,
        "sentiment": sentiment,
        "topic": predict_topic(text),
        "role": predict_role(text, num_questions),
    }
