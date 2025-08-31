import pandas as pd
import re
import zipfile
import os

# --- Cleaning function ---
def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r"http\S+|www\S+|https\S+", "", str(text))
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

# --- Load datasets ---
debates = pd.read_csv("data/2020 Debate Transcripts.csv", encoding="ISO-8859-1")
un_debates = pd.read_csv("data/un-general-debates.csv", encoding="ISO-8859-1")

# --- Clean Debate Dataset ---
debates_clean = debates.copy()
debates_clean.rename(columns={
    "Discourse w/names": "discourse_with_names",
    "Discourse w/o names": "discourse_without_names",
    "SpeakerInfo": "speaker_info",
    "Original Row information": "row_info"
}, inplace=True)

debates_clean["discourse_with_names"] = debates_clean["discourse_with_names"].apply(clean_text)
debates_clean["discourse_without_names"] = debates_clean["discourse_without_names"].apply(clean_text)

# --- Clean UN Dataset ---
un_clean = un_debates.copy()
un_clean["text"] = un_clean["text"].apply(clean_text)

# --- Save Cleaned Files ---
debates_clean.to_csv("debates_clean.csv", index=False)
un_clean.to_csv("un_general_clean.csv", index=False)

# --- Zip Both Files ---
with zipfile.ZipFile("cleaned_datasets.zip", "w") as zf:
    zf.write("debates_clean.csv")
    zf.write("un_general_clean.csv")

print("✅ Cleaned files saved and zipped as cleaned_datasets.zip")
