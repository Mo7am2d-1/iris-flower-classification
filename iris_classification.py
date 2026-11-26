"""Iris flower classification: compare KNN, SVM and a decision tree.

Dataset: classic Iris dataset (150 flowers, 3 species).
We train three simple models and compare their accuracy on a held-out
test split, then plot a confusion matrix for the best one.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# 1. Load the data
df = pd.read_csv("iris.csv")
print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")
print(df["species"].value_counts(), "\n")

X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = df["species"]

# 2. Split: 80% train, 20% test (fixed seed so results are reproducible)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Train three models and compare
models = {
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "SVM (RBF kernel)": SVC(kernel="rbf", random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    results[name] = acc
    print(f"{name:18s} test accuracy: {acc:.3f}")

best_name = max(results, key=results.get)
print(f"\nBest model: {best_name}")

# 4. Confusion matrix for the best model
best_model = models[best_name]
cm = confusion_matrix(y_test, best_model.predict(X_test), labels=best_model.classes_)
disp = ConfusionMatrixDisplay(cm, display_labels=best_model.classes_)
disp.plot(cmap="Blues")
plt.title(f"Iris - confusion matrix ({best_name})")
plt.tight_layout()
plt.savefig("results.png", dpi=120)
print("Saved results.png")
