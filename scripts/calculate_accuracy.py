import pandas as pd

# Load the predictions CSV
df = pd.read_csv('predictions.csv')

# Remove rows where prediction failed
df_clean = df[df['predicted_emotion'] != 'ERROR']

# Calculate accuracy
correct = (df_clean['true_emotion'] == df_clean['predicted_emotion']).sum()
total = len(df_clean)
accuracy = correct / total if total > 0 else 0

# Print summary
print(f"✅ Total Samples: {total}")
print(f"✅ Correct Predictions: {correct}")
print(f"🎯 Accuracy: {accuracy * 100:.2f}%")
