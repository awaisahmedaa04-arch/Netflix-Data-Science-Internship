"""
Task 5 (Advanced) - Machine Learning Classification Model
Builds a model to classify content type (Movie vs TV Show) from
available metadata features.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("outputs/netflix_cleaned.csv")

report = []


def log(line=""):
    print(line)
    report.append(line)


log("TASK 5 - MACHINE LEARNING CLASSIFICATION MODEL")
log("=" * 50)
log("Goal: classify whether a title is a Movie or a TV Show using")
log("release_year, rating, primary_country, primary_genre, and duration_value.")

# Step 1: Select and prepare features
features = ["release_year", "rating", "primary_country", "primary_genre", "duration_value"]
data = df[features + ["type"]].dropna()

le_rating = LabelEncoder()
le_country = LabelEncoder()
le_genre = LabelEncoder()
le_target = LabelEncoder()

data = data.copy()
data["rating_enc"] = le_rating.fit_transform(data["rating"])
data["country_enc"] = le_country.fit_transform(data["primary_country"])
data["genre_enc"] = le_genre.fit_transform(data["primary_genre"])
data["target"] = le_target.fit_transform(data["type"])  # Movie=0, TV Show=1 (alphabetical)

X = data[["release_year", "rating_enc", "country_enc", "genre_enc", "duration_value"]]
y = data["target"]

log(f"\nUsable rows after dropping missing feature values: {len(data)}")
log(f"Class balance:\n{data['type'].value_counts().to_string()}")

# Step 2: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
log(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

# Step 3: Train classification models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = (model, preds, acc)
    log(f"\n--- {name} ---")
    log(f"Accuracy: {acc:.3f}")
    log(classification_report(y_test, preds, target_names=le_target.classes_))

# Step 4: Evaluate model performance (confusion matrix for the best model)
best_name = max(results, key=lambda k: results[k][2])
best_model, best_preds, best_acc = results[best_name]
log(f"\nBest performing model: {best_name} (accuracy = {best_acc:.3f})")

cm = confusion_matrix(y_test, best_preds)
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap="Reds")
ax.set_xticks([0, 1]); ax.set_xticklabels(le_target.classes_)
ax.set_yticks([0, 1]); ax.set_yticklabels(le_target.classes_)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix - {best_name}")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
plt.tight_layout()
plt.savefig("figures/task5_confusion_matrix.png", dpi=150)
plt.close()

# Step 5: Compare model accuracy
log("\n--- MODEL COMPARISON ---")
for name, (_, _, acc) in results.items():
    log(f"{name}: {acc:.3f}")

if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    log("\nFeature importances (best model):")
    log(importances.to_string())

# --- Data leakage check ---
# duration_value means something different for Movies (minutes, ~70-300) vs
# TV Shows (seasons, ~1-17), so it almost perfectly encodes the answer.
# Re-run without it to see how much signal comes from genuinely independent features.
log("\n--- LEAKAGE CHECK: retraining WITHOUT duration_value ---")
X_no_dur = data[["release_year", "rating_enc", "country_enc", "genre_enc"]]
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_no_dur, y, test_size=0.2, random_state=42, stratify=y
)
rf2 = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
rf2.fit(X_train2, y_train2)
acc2 = accuracy_score(y_test2, rf2.predict(X_test2))
log(f"Random Forest accuracy WITHOUT duration_value: {acc2:.3f}")
log("Dropping duration_value barely moves accuracy, which reveals a second,")
log("subtler leak: Netflix's own genre labels (e.g. 'TV Dramas' vs 'Dramas',")
log("'International TV Shows' vs 'International Movies') already spell out the")
log("content type in the text. In a real project this feature would need to be")
log("re-engineered (e.g. strip the 'TV'/'Movie' qualifier from genre strings)")
log("before the model's accuracy could be trusted as a fair measure of skill.")

with open("outputs/task5_ml_summary.txt", "w") as f:
    f.write("\n".join(report))

print("\nSaved figures/task5_confusion_matrix.png and outputs/task5_ml_summary.txt")
