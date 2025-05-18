import os
import glob
import librosa
import pandas as pd

import numpy as np
import librosa

def extract_features(y, sr):
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = mfcc.mean(axis=1)  # shape (13,)
        return mfcc_mean
    except Exception as e:
        print(f"⚠️ Error extracting features: {e}")
        raise

# --- Emotion mapping from RAVDESS ---
emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# --- Collect features ---
features = []

base_path = r"C:\Users\Sania\Documents\GitHub\voice-mood-detector\data\archive"
output_csv = r"C:\Users\Sania\Documents\GitHub\voice-mood-detector\data\features.csv"
wav_files = glob.glob(os.path.join(base_path, "Actor_*", "*.wav"))

print(f"🎧 Found {len(wav_files)} .wav files")

for file_path in wav_files:
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = mfcc.mean(axis=1)  # Take mean of each coefficient

        # Extract emotion from filename (3rd group)
        filename = os.path.basename(file_path)
        parts = filename.split("-")
        emotion_code = parts[2]
        emotion = emotion_map.get(emotion_code, "unknown")

        # Store features and label
        features.append({
            "file": filename,
            "emotion": emotion,
            **{f"mfcc_{i+1}": mfcc_mean[i] for i in range(len(mfcc_mean))}
        })

    except Exception as e:
        print(f"⚠️ Skipped {file_path}: {e}")

# --- Save to DataFrame and CSV ---
df = pd.DataFrame(features)
print(f"\n✅ Extracted features from {len(df)} files")

df.to_csv(output_csv, index=False)
print("📁 Saved to data/features.csv")
