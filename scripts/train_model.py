import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Load feature data
df = pd.read_csv(r"C:\GitHub\voice-mood-detector\data\archive\features.csv")

# 2. Prepare X (features) and y (labels)
X = df.drop(columns=["file", "emotion"])
y = df["emotion"]

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
print("\n🧠 Classification Report:")
print(classification_report(y_test, y_pred))
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# 6. Save model
joblib.dump(model, r"C:\GitHub\voice-mood-detector\models\mood_classifier.pkl")
print("💾 Model saved to models/mood_classifier.pkl")
