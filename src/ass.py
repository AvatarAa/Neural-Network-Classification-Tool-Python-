import math
import matplotlib.pyplot as plt
import networkx as nx

# ---------- Helper functions ----------

def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def forward_pass(x, W):
    """
    x: (x1, x2, x3)
    W: (W1..W8)
    returns dict of all activations and weighted sums
    """
    x1, x2, x3 = x
    W1, W2, W3, W4, W5, W6, W7, W8 = W

    z1 = W1 * x1 + W2 * x2 + W3 * x3
    a1 = sigmoid(z1)

    z2 = W4 * x1 + W5 * x2 + W6 * x3
    a2 = sigmoid(z2)

    z3 = W7 * a1 + W8 * a2
    o = sigmoid(z3)

    return {
        "z1": z1, "a1": a1,
        "z2": z2, "a2": a2,
        "z3": z3, "o": o
    }


def grads_one_example(x, y, W):
    """
    Compute dL/dW for one training example using
    mean-squared error: L = 0.5 * (o - y)^2
    """
    out = forward_pass(x, W)
    a1, a2, o = out["a1"], out["a2"], out["o"]
    x1, x2, x3 = x
    W1, W2, W3, W4, W5, W6, W7, W8 = W

    # dL/do for MSE
    dL_do = o - y
    # derivative of sigmoid
    do_dz3 = o * (1 - o)
    dL_dz3 = dL_do * do_dz3

    # Gradients for output weights
    dL_dW7 = dL_dz3 * a1
    dL_dW8 = dL_dz3 * a2

    # Hidden neuron 1
    dz3_da1 = W7
    da1_dz1 = a1 * (1 - a1)
    dL_dz1 = dL_dz3 * dz3_da1 * da1_dz1
    dL_dW1 = dL_dz1 * x1
    dL_dW2 = dL_dz1 * x2
    dL_dW3 = dL_dz1 * x3

    # Hidden neuron 2
    dz3_da2 = W8
    da2_dz2 = a2 * (1 - a2)
    dL_dz2 = dL_dz3 * dz3_da2 * da2_dz2
    dL_dW4 = dL_dz2 * x1
    dL_dW5 = dL_dz2 * x2
    dL_dW6 = dL_dz2 * x3

    return [dL_dW1, dL_dW2, dL_dW3,
            dL_dW4, dL_dW5, dL_dW6,
            dL_dW7, dL_dW8]


def train_one_epoch(dataset, W, lr=0.01):
    """
    dataset: list of (x, y) pairs
    W: current weights
    lr: learning rate
    returns updated weights after 1 epoch
    """
    W = list(W)
    for x, y in dataset:
        grads = grads_one_example(x, y, W)
        W = [w - lr * g for w, g in zip(W, grads)]
    return W


# ---------- 1. Original example (same role as Canvas & playground) ----------

# You can replace these with the actual numbers from your Canvas example
X1 = 40.0   # Age
X2 = 0.2    # Debt ratio
X3 = 60.0   # Income

# Weights W1..W8 (replace with your exact slide values if different)
W = [
    0.05,  # W1
    0.10,  # W2
    0.20,  # W3
    0.30,  # W4
    0.25,  # W5
    0.15,  # W6
    0.40,  # W7
    0.45   # W8
]

x_original = (X1, X2, X3)

print("=== Q1: Forward pass for original example ===")
out_orig = forward_pass(x_original, W)
print(f"z1 = {out_orig['z1']:.4f}, H1 = {out_orig['a1']:.6f}")
print(f"z2 = {out_orig['z2']:.4f}, H2 = {out_orig['a2']:.6f}")
print(f"z3 = {out_orig['z3']:.4f}, Output (O3) = {out_orig['o']:.6f}")
print()

# ---------- 2. Normalized inputs (for the normalization question) ----------

# You can adjust the max values if your prof used different ranges
max_age = 100.0
max_debt = 1.0       # already between 0 and 1
max_income = 100.0

x_normalized = (
    X1 / max_age,
    X2 / max_debt,
    X3 / max_income
)

print("=== Forward pass with NORMALIZED inputs ===")
out_norm = forward_pass(x_normalized, W)
print(f"z1 = {out_norm['z1']:.4f}, H1 = {out_norm['a1']:.6f}")
print(f"z2 = {out_norm['z2']:.4f}, H2 = {out_norm['a2']:.6f}")
print(f"z3 = {out_norm['z3']:.4f}, Output (O3) = {out_norm['o']:.6f}")
print()

# ---------- 3. Add a second training example & 1 epoch of training ----------

# Second training example (you can change these)
x2 = (25.0, 0.5, 30.0)
y1 = 1.0   # label for original example (approved)
y2 = 0.0   # label for second example (not approved)

dataset = [
    (x_original, y1),
    (x2, y2)
]

print("=== Weights BEFORE 1 epoch of training ===")
for i, w in enumerate(W, start=1):
    print(f"W{i} = {w:.6f}")

W_updated = train_one_epoch(dataset, W, lr=0.01)

print("\n=== Weights AFTER 1 epoch of training ===")
for i, w in enumerate(W_updated, start=1):
    print(f"W{i} = {w:.6f}")

print("\n=== Output on original example AFTER training ===")
out_after = forward_pass(x_original, W_updated)
print(f"Output (O3) after training = {out_after['o']:.6f}")
print()

# ---------- 4. Draw the ANN diagram and save as PNG ----------

G = nx.DiGraph()

# Nodes
G.add_node("X1\n(Age)", color="lightblue")
G.add_node("X2\n(Debt)", color="lightblue")
G.add_node("X3\n(Income)", color="lightblue")

G.add_node("H1", color="orange")
G.add_node("H2", color="orange")

G.add_node("O3\n(Output)", color="lightgreen")

# Edges with current (original) weights
W1, W2, W3, W4, W5, W6, W7, W8 = W

G.add_edge("X1\n(Age)", "H1", weight=W1)
G.add_edge("X2\n(Debt)", "H1", weight=W2)
G.add_edge("X3\n(Income)", "H1", weight=W3)

G.add_edge("X1\n(Age)", "H2", weight=W4)
G.add_edge("X2\n(Debt)", "H2", weight=W5)
G.add_edge("X3\n(Income)", "H2", weight=W6)

G.add_edge("H1", "O3\n(Output)", weight=W7)
G.add_edge("H2", "O3\n(Output)", weight=W8)

# Positions
pos = {
    "X1\n(Age)": (-2, 1),
    "X2\n(Debt)": (-2, 0),
    "X3\n(Income)": (-2, -1),
    "H1": (0, 0.7),
    "H2": (0, -0.7),
    "O3\n(Output)": (2, 0),
}

colors = [G.nodes[n]["color"] for n in G.nodes()]
nx.draw(G, pos, with_labels=True, node_color=colors,
        node_size=3000, font_size=9)

edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

plt.title("Loan Approval ANN (3 → 2 → 1)")
plt.margins(0.2)   # avoids the tight_layout warning
plt.savefig("loan_ann.png", dpi=200)
plt.show()

print("Diagram saved as loan_ann.png – you can insert this image into your report.")
