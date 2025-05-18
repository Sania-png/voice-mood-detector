# app.py
from flask import Flask, request, jsonify
import os
import librosa
import numpy as np
import joblib

# Import your scripts
from scripts.load_audio import load_audio  # modify import if needed
from scripts.extract_features import extract_features

app = Flask(__name__)

# Load trained model
model = joblib.load(r"C:\Users\Sania\Documents\GitHub\voice-mood-detector\models\mood_classifier.pkl")


@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    file = request.files['audio']
    file_path = os.path.join('temp', file.filename)
    os.makedirs('temp', exist_ok=True)
    file.save(file_path)

    try:
        # Load and preprocess audio
        y, sr = load_audio(file_path)
        features = extract_features(y, sr)
        features = np.expand_dims(features, axis=0)  # shape for model input

        # Predict mood
        prediction = model.predict(features)[0]

        return jsonify({'mood': prediction})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        os.remove(file_path)


if __name__ == '__main__':
    app.run(debug=True)