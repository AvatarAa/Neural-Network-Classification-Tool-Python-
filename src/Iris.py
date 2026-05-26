


import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss, classification_report

# ---------- 1. Load and split the data ----------

iris = load_iris()
X = iris.data                     # 4 inputs: sepal_length, sepal_width, petal_length, petal_width
y = iris.target                   # 0 = setosa, 1 = versicolor, 2 = virginica
class_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Standardize (important for MLP)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Classes:", class_names)
print("Train size:", X_train.shape[0], " Test size:", X_test.shape[0])
print("-" * 60)

# ---------- 2. Define the experiments you want to run ----------

# Each entry: (name, hidden_layer_sizes, learning_rate)
experiments = [
    ("No hidden layer (linear-ish)", (),      0.01),
    ("1 hidden layer (4), lr=0.01",   (4,),   0.01),
    ("1 hidden layer (4), lr=0.05",   (4,),   0.05),
    ("1 hidden layer (4), lr=0.10",   (4,),   0.10),
    ("2 hidden layers (4,4), lr=0.05", (4,4), 0.05),
]

results = []

# ---------- 3. Run all experiments ----------

for name, hidden, lr in experiments:
    print(f"\n===== {name}  |  hidden={hidden}  |  lr={lr} =====")

    # If hidden == () this is basically no hidden layer
    clf = MLPClassifier(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        learning_rate_init=lr,
        max_iter=800,
        random_state=42
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_proba)

    print(f"Loss     : {loss:.4f}")
    print(f"Accuracy : {acc * 100:.2f}%")

    # Save for summary table
    results.append((name, hidden, lr, loss, acc))

# ---------- 4. Print a nice summary table ----------

print("\n" + "=" * 60)
print("SUMMARY TABLE (copy this into your assignment)")
print("=" * 60)
print(f"{'Architecture':35s} {'Hidden':12s} {'LR':6s} {'Loss':8s} {'Acc%':6s}")
for name, hidden, lr, loss, acc in results:
    h_str = str(hidden) if hidden else "None"
    print(f"{name:35s} {h_str:12s} {lr:<6.2f} {loss:<8.4f} {acc*100:6.2f}")

# ---------- 5. Show per-class performance for the best model (2 layers) ----------

print("\nPer-class report for best model (2 hidden layers, lr=0.05):")
best_exp = experiments[-1]
_, hidden, lr = best_exp

best_clf = MLPClassifier(
    hidden_layer_sizes=hidden,
    activation="relu",
    solver="adam",
    learning_rate_init=lr,
    max_iter=800,
    random_state=42
)
best_clf.fit(X_train, y_train)
best_pred = best_clf.predict(X_test)

print(classification_report(y_test, best_pred, target_names=class_names, digits=3))
