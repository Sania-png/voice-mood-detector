import os
import librosa
import matplotlib.pyplot as plt
import librosa.display  # Make sure to import this for specshow

# Use the direct path to your audio file
# Replace this with your copied path
full_path = r"C:\GitHub\voice-mood-detector\data\archive\Actor_01\03-01-01-01-01-01-01.wav"  # The 'r' prefix makes it a raw string to handle backslashes

# Check if the file exists before trying to load it
if not os.path.isfile(full_path):
    raise FileNotFoundError(f"Audio file not found: {full_path}")

# Load the audio file
try:
    y, sr = librosa.load(full_path)

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Plot MFCCs
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"Error processing audio file: {e}")