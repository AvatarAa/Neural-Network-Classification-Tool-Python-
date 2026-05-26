# ==============================================
#Iris Neural Network Experiments
# Works with YOUR FILE: Iris.csv
# ==============================================

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# LOAD YOUR FILE
# --------------------------------------------------
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "Iris.csv"
df = pd.read_csv(DATA_PATH)   # <-- USE YOUR FILE HERE
print("\nLoaded Dataset:\n")
print(df.head())

# --------------------------------------------------
# PREPROCESSING
# --------------------------------------------------
X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
y = df["Species"]

# Encode labels (setosa → 0, versicolor → 1, virginica → 2)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Normalize numeric inputs
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# --------------------------------------------------
# EXPERIMENT FUNCTION
# --------------------------------------------------
def run_experiment(hidden_layers, lr):
    print(f"\n=== Running: Hidden Layers = {hidden_layers}, LR = {lr} ===")

    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        learning_rate_init=lr,
        solver="adam",
        max_iter=200,
        random_state=0
    )

    clf.fit(X_train, y_train)

    # LOSS CURVE
    plt.figure(figsize=(7,4))
    plt.plot(clf.loss_curve_, linewidth=2)
    plt.title(f"Loss Curve — Layers={hidden_layers}, LR={lr}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

    # ACCURACY
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc*100:.2f}%")

    return acc

# --------------------------------------------------
# RUN REQUIRED Q2 EXPERIMENTS
# --------------------------------------------------

# 1️⃣ No Hidden Layer
run_experiment((), 0.05)

# 2️⃣ One Hidden Layer (4 neurons)
run_experiment((4,), 0.05)

# 3️⃣ Two Hidden Layers (4,4)
run_experiment((4,4), 0.05)

# 4️⃣ Learning Rate Experiments
for lr_value in [0.01, 0.05, 0.1]:
    run_experiment((4,), lr_value)

