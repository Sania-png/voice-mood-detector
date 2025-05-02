import librosa

test_file = r"C:\GitHub\voice-mood-detector\data\archive\Actor_01\03-01-01-01-01-01-01.wav"

try:
    y, sr = librosa.load(test_file, sr=None)
    print("✅ Loaded test file")
    print("Duration (sec):", librosa.get_duration(y=y, sr=sr))
except Exception as e:
    print(f"⚠️ Could not load test file: {e}")