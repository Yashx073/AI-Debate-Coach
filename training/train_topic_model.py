import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load standardized data
df = pd.read_csv("data/debates_standardized.csv")

# Drop missing topics
df = df[df["topic"].notna()]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["topic"], test_size=0.2, random_state=42
)

# Pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
    ("clf", LogisticRegression(max_iter=200))
])

# Train
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save
joblib.dump(model, "models/topic_model.pkl")
print("✅ Topic model saved to models/topic_model.pkl")
