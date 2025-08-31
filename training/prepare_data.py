import pandas as pd
from sklearn.model_selection import train_test_split

# --- Load cleaned debate dataset ---
debate_df = pd.read_csv("data/debates_clean.csv")

# --- Standardize column names ---
debate_df.rename(columns={
    "discourse_without_names": "text",
    "Party": "party",
    "Topic": "topic",
    "Speaker": "speaker"
}, inplace=True)

# --- Drop empty rows (if any) ---
debate_df.dropna(subset=["text"], inplace=True)

# --- Save standardized version (so all models use same file) ---
debate_df.to_csv("data/debates_standardized.csv", index=False)

# --- Train/Test Split ---
train_df, test_df = train_test_split(
    debate_df,
    test_size=0.2,
    random_state=42,
    stratify=debate_df["party"]
)

# --- Save train/test datasets ---
train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

print("✅ Data prepared and saved")
print(f"Training samples: {len(train_df)}")
print(f"Testing samples: {len(test_df)}")
