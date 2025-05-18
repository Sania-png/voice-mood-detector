import librosa

def load_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        print("✅ Loaded audio file")
        print("Duration (sec):", librosa.get_duration(y=y, sr=sr))
        return y, sr
    except Exception as e:
        print(f"⚠️ Could not load audio file: {e}")
        raise