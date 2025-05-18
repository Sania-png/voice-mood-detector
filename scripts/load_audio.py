import librosa

def load_audio(file_path):
    """
    Loads a WAV audio file using librosa.

    Args:
        file_path (str): Path to the WAV file.

    Returns:
        y (np.ndarray): Audio time series
        sr (int): Sampling rate
    """
    try:
        y, sr = librosa.load(file_path, sr=None)
        print("✅ Loaded test file")
        print("Duration (sec):", librosa.get_duration(y=y, sr=sr))
        return y, sr
    except Exception as e:
        print(f"⚠️ Could not load file: {e}")
        raise e  # Re-raise the error so calling code can handle it
