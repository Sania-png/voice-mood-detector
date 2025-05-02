import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the features CSV
df = pd.read_csv(r"C:\GitHub\voice-mood-detector\data\features.csv")

# Prepare log directory
os.makedirs("logs", exist_ok=True)
log_path = "logs/balance_report.txt"

with open(log_path, "w") as log:
    # Count emotions
    emotion_counts = df['emotion'].value_counts()
    total = len(df)

    log.write("Emotion Distribution:\n\n")
    log.write(emotion_counts.to_string())
    log.write("\n\n")

    print("Emotion Distribution:\n")
    print(emotion_counts)
    print()

    # Check for imbalance
    threshold = 0.05 * total
    rare_emotions = emotion_counts[emotion_counts < threshold]

    if not rare_emotions.empty:
        message = (
            "Unbalanced dataset detected!\n"
            "The following emotions have fewer than 5% of total samples:\n\n"
            f"{rare_emotions.to_string()}\n\n"
            "Consider oversampling or excluding these classes."
        )
        print(message)
        log.write(message)
    else:
        print("Dataset appears balanced.")
        log.write("Dataset appears balanced.\n")

# Bar Plot
plt.figure(figsize=(10, 6))
bars = plt.bar(emotion_counts.index, emotion_counts.values, color='skyblue')
for bar, count in zip(bars, emotion_counts):
    percent = (count / total) * 100
    plt.text(bar.get_x() + bar.get_width()/2, count + 2, f'{percent:.1f}%', ha='center', fontsize=9)
plt.title("Emotion Distribution (Bar Chart)")
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

# Pie Chart
plt.figure(figsize=(8, 8))
plt.pie(emotion_counts, labels=emotion_counts.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
plt.title("Emotion Distribution (Pie Chart)")
plt.axis('equal')  # Equal aspect ratio ensures pie is circular
plt.tight_layout()
plt.show()