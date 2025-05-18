import streamlit as st
import numpy as np
import joblib
import tempfile
import sys
import os

# Ensure the parent directory is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from load_audio import load_audio
from extract_features import extract_features

# Load trained model
import os

model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'mood_classifier.pkl')
model_path = os.path.abspath(model_path)
model = joblib.load(model_path)

st.title("🎤 Speech Mood Detector")

# Upload WAV file
uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file:
    # Show the audio player
    st.audio(uploaded_file, format='audio/wav')

    # Save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Preprocess and extract features
    try:
        y, sr = load_audio(tmp_path)
        features = extract_features(y, sr)
        features = np.expand_dims(features, axis=0)

        # Predict
        prediction = model.predict(features)[0]
        st.success(f"💬 Detected Mood: **{prediction}**")

    except Exception as e:
        st.error(f"Failed to process audio: {e}")


