import os
import requests
import csv

# Define paths
folder_path = r'C:\Users\Sania\Documents\GitHub\voice-mood-detector\data\archive'
url = 'http://127.0.0.1:5000/predict'
output_file = 'predictions.csv'

# Emotion code mapping from RAVDESS filename (3rd value)
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

# Create CSV and write header
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['filename', 'true_emotion', 'predicted_emotion'])

    # Walk through all Actor folders and WAV files
    for root, _, files in os.walk(folder_path):
        wav_files = [f for f in files if f.endswith('.wav')]
        for filename in wav_files:
            file_path = os.path.join(root, filename)

            # Extract true emotion from filename (3rd field)
            try:
                parts = filename.split("-")
                emotion_code = parts[2]
                true_emotion = emotion_map.get(emotion_code, "unknown")
            except Exception:
                true_emotion = "unknown"

            # Send to API
            try:
                with open(file_path, 'rb') as f:
                    response = requests.post(url, files={'audio': f})
                prediction = response.json().get('mood', 'ERROR')
            except Exception as e:
                prediction = 'ERROR'

            # Write result
            writer.writerow([filename, true_emotion, prediction])
            print(f"{filename}: true={true_emotion}, predicted={prediction}")

