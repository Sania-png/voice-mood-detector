import pandas as pd
import os  # Add this
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler  # Add this
import joblib

# --- Step 1: Load the dataset ---
df = pd.read_csv(r"C:\GitHub\voice-mood-detector\data\features.csv")

X = df.drop(columns=["file", "emotion"])
y = df["emotion"]

# --- Step 2: Split data with stratification ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Step 3: Feature Scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Step 4: Define models with improved parameters ---
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM (Linear Kernel)": SVC(kernel='linear', C=1, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=2000, solver='saga')  # Increased iterations
}

# --- Step 5: Train and evaluate ---
best_model = None
best_accuracy = 0

for name, model in models.items():
    print(f"\n🔍 Training: {name}")

    # Use scaled data for non-tree models
    if name in ["SVM (Linear Kernel)", "KNN", "Logistic Regression"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc * 100:.2f}%")
    print("📋 Classification Report:")
    print(classification_report(y_test, y_pred))

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

# --- Step 6: Save the best model properly ---
os.makedirs("models", exist_ok=True)  # Create directory if needed
joblib.dump(best_model, "models/best_model.pkl")  # Fixed filename/path
print("\n💾 Best model saved to models/best_model.pkl")