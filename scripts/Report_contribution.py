import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Set the style for plots
plt.style.use('fivethirtyeight')
sns.set_palette("deep")


def analyze_feature_importance():
    """Analyze and visualize the contribution of each feature to the model's performance"""
    print("📊 Analyzing Feature Contributions...")

    # --- Step 1: Find and load the dataset ---
    # Try different possible locations for the features.csv file
    possible_paths = [
        "data/features.csv",
        "features.csv",
        "../data/features.csv",
        "scripts/data/features.csv",
        os.path.join(os.path.dirname(__file__), r"C:\GitHub\voice-mood-detector\data\archive\features.csv")
    ]

    df = None
    for path in possible_paths:
        try:
            if os.path.exists(path):
                print(f"✅ Found features file at: {path}")
                df = pd.read_csv(path)
                break
        except Exception:
            continue

    if df is None:
        print("❌ Error: Could not find features.csv file. Please specify the correct path.")
        print("Searched in the following locations:")
        for path in possible_paths:
            print(f"  - {os.path.abspath(path)}")
        return

    # Ensure results directory exists
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"✅ Created results directory at {os.path.abspath(results_dir)}")

    # Ensure models directory exists for saving/loading models
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"✅ Created models directory at {os.path.abspath(models_dir)}")

    # --- Step 2: Prepare data ---
    X = df.drop(columns=["file", "emotion"] if "file" in df.columns and "emotion" in df.columns else
    [col for col in df.columns if col in ["file", "emotion"]])

    if "emotion" not in df.columns:
        print("❌ Error: The dataset does not contain an 'emotion' column.")
        return

    y = df["emotion"]
    feature_names = X.columns

    # --- Step 3: Split data ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Step 4: Load the best model ---
    best_model_path = os.path.join(models_dir, "best_model.pkl")
    try:
        if os.path.exists(best_model_path):
            best_model = joblib.load(best_model_path)
            print(f"✅ Loaded best model from {best_model_path}")
        else:
            print("⚠️ Best model not found, training a Random Forest model instead...")
            best_model = RandomForestClassifier(n_estimators=100, random_state=42)
            best_model.fit(X_train, y_train)
    except Exception as e:
        print(f"⚠️ Error loading model: {str(e)}. Training a new Random Forest model instead...")
        best_model = RandomForestClassifier(n_estimators=100, random_state=42)
        best_model.fit(X_train, y_train)

    # --- Step 5: Calculate feature importance ---
    # Method 1: Built-in feature importance (if available)
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]

        # Plot feature importance
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importance (Built-in Method)', fontsize=18)
        plt.bar(range(X.shape[1]), importances[indices], align='center')
        plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()

        feature_importance_path = os.path.join(results_dir, 'feature_importance_builtin.png')
        plt.savefig(feature_importance_path, dpi=300, bbox_inches='tight')
        print(f"📸 Saved feature importance chart to {feature_importance_path}")

        # Print top 10 features
        print("\n🔝 Top 10 Most Important Features (Built-in Method):")
        for i in range(10):
            if i < len(indices):
                print(f"{i + 1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

    # Method 2: Permutation importance (works for any model)
    try:
        result = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42)
        perm_importances = result.importances_mean
        perm_indices = np.argsort(perm_importances)[::-1]

        # Plot permutation importance
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importance (Permutation Method)', fontsize=18)
        plt.bar(range(X.shape[1]), perm_importances[perm_indices], align='center')
        plt.xticks(range(X.shape[1]), [feature_names[i] for i in perm_indices], rotation=90)
        plt.tight_layout()

        perm_importance_path = os.path.join(results_dir, 'feature_importance_permutation.png')
        plt.savefig(perm_importance_path, dpi=300, bbox_inches='tight')
        print(f"📸 Saved permutation importance chart to {perm_importance_path}")

        # Print top 10 features
        print("\n🔝 Top 10 Most Important Features (Permutation Method):")
        for i in range(10):
            if i < len(perm_indices):
                print(f"{i + 1}. {feature_names[perm_indices[i]]}: {perm_importances[perm_indices[i]]:.4f}")
    except Exception as e:
        print(f"⚠️ Error calculating permutation importance: {str(e)}")

    # --- Step 6: Correlation analysis ---
    try:
        # Calculate correlation with target (point-biserial for binary targets)
        # For multiclass, we'll create a heatmap of feature correlations
        plt.figure(figsize=(14, 12))
        correlation_matrix = X.corr()
        mask = np.triu(correlation_matrix)
        sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm',
                    linewidths=0.5, vmin=-1, vmax=1)
        plt.title('Feature Correlation Matrix', fontsize=18)
        plt.tight_layout()

        correlation_path = os.path.join(results_dir, 'feature_correlation.png')
        plt.savefig(correlation_path, dpi=300, bbox_inches='tight')
        print(f"📸 Saved feature correlation matrix to {correlation_path}")
    except Exception as e:
        print(f"⚠️ Error creating correlation matrix: {str(e)}")

    # --- Step 7: Feature contribution by emotion ---
    if hasattr(best_model, 'feature_importances_'):
        try:
            # Get unique emotions
            emotions = y.unique()

            # Create a figure with subplots for each emotion
            fig, axes = plt.subplots(len(emotions), 1, figsize=(12, 5 * len(emotions)))

            for i, emotion in enumerate(emotions):
                # Get indices for this emotion
                emotion_indices = y_train == emotion

                # Train a model just for this emotion (one-vs-rest)
                emotion_model = RandomForestClassifier(n_estimators=50, random_state=42)
                y_binary = y_train.map(lambda x: 1 if x == emotion else 0)
                emotion_model.fit(X_train, y_binary)

                # Get feature importances
                emotion_importances = emotion_model.feature_importances_
                emotion_indices_sorted = np.argsort(emotion_importances)[::-1][:10]  # Top 10

                # Plot
                ax = axes[i] if len(emotions) > 1 else axes
                ax.bar(range(10), emotion_importances[emotion_indices_sorted], align='center')
                ax.set_xticks(range(10))
                ax.set_xticklabels([feature_names[j] for j in emotion_indices_sorted], rotation=45, ha='right')
                ax.set_title(f'Top 10 Features for Emotion: {emotion}', fontsize=16)
                ax.set_ylabel('Importance')

            plt.tight_layout()

            emotion_importance_path = os.path.join(results_dir, 'feature_importance_by_emotion.png')
            plt.savefig(emotion_importance_path, dpi=300, bbox_inches='tight')
            print(f"📸 Saved emotion-specific feature importance to {emotion_importance_path}")
        except Exception as e:
            print(f"⚠️ Error creating emotion-specific feature importance: {str(e)}")

    # --- Step 8: Generate key findings for report ---
    print("\n📝 Key Findings for Final Report:")

    if 'perm_indices' in locals() and len(perm_indices) > 0:
        print("• The most discriminative audio features for emotion classification are: " +
              ", ".join([feature_names[perm_indices[i]] for i in range(min(3, len(perm_indices)))]))
    elif 'indices' in locals() and len(indices) > 0:
        print("• The most discriminative audio features for emotion classification are: " +
              ", ".join([feature_names[indices[i]] for i in range(min(3, len(indices)))]))

    if hasattr(best_model, 'feature_importances_') and 'emotions' in locals():
        emotion_specific_features = {}
        for emotion in emotions:
            emotion_indices = y_train == emotion
            emotion_model = RandomForestClassifier(n_estimators=50, random_state=42)
            y_binary = y_train.map(lambda x: 1 if x == emotion else 0)
            emotion_model.fit(X_train, y_binary)
            emotion_importances = emotion_model.feature_importances_
            emotion_indices_sorted = np.argsort(emotion_importances)[::-1][:3]
            emotion_specific_features[emotion] = [feature_names[j] for j in emotion_indices_sorted]

        # Find an interesting emotion to highlight
        highlight_emotion = emotions[0]  # Default to first emotion
        print(f"• For detecting '{highlight_emotion}' emotions specifically, the key features are: " +
              ", ".join(emotion_specific_features[highlight_emotion]))

    # Find highly correlated features
    if 'correlation_matrix' in locals():
        high_corr_threshold = 0.8
        high_corr_pairs = []
        for i in range(len(feature_names)):
            for j in range(i + 1, len(feature_names)):
                if abs(correlation_matrix.iloc[i, j]) > high_corr_threshold:
                    high_corr_pairs.append((feature_names[i], feature_names[j], correlation_matrix.iloc[i, j]))

        if high_corr_pairs:
            print("• Several features show high correlation and may be redundant, including: " +
                  f"{high_corr_pairs[0][0]} and {high_corr_pairs[0][1]} (r={high_corr_pairs[0][2]:.2f})")
        else:
            print("• The feature set shows minimal redundancy, with most features providing unique information")


if __name__ == "__main__":
    analyze_feature_importance()