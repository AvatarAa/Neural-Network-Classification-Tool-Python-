import math
import matplotlib.pyplot as plt
import networkx as nx

# -----------------------------
# 1. Sigmoid function
# -----------------------------
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

# -----------------------------
# 2. INPUT VALUES (Loan Example)
# -----------------------------
X1 = 40    # Age
X2 = 0.2   # Debt ratio
X3 = 60    # Income

# -----------------------------
# 3. WEIGHTS (from Canvas Example or your values)
# -----------------------------
# Hidden neuron H1
W1 = 0.05
W2 = 0.10
W3 = 0.20

# Hidden neuron H2
W4 = 0.30
W5 = 0.25
W6 = 0.15

# Output neuron
W7 = 0.40
W8 = 0.45

# -----------------------------
# 4. FORWARD PASS
# -----------------------------

# Hidden neuron H1
F = W1*X1 + W2*X2 + W3*X3
H1 = sigmoid(F)

# Hidden neuron H2
G = W4*X1 + W5*X2 + W6*X3
H2 = sigmoid(G)

# Output
F1 = W7*H1 + W8*H2
O3 = sigmoid(F1)

# Print results
print("=== Forward Pass Results ===")
print(f"H1 weighted sum F = {F:.4f}  →  H1 = {H1:.6f}")
print(f"H2 weighted sum G = {G:.4f}  →  H2 = {H2:.6f}")
print(f"Output weighted sum F1 = {F1:.4f}  →  Output (Loan Approval Probability) = {O3:.6f}")

# -----------------------------
# 5. DRAW DIAGRAM
# -----------------------------
G = nx.DiGraph()

# Add nodes
G.add_node("X1\n(Age)", color='lightblue')
G.add_node("X2\n(Debt Ratio)", color='lightblue')
G.add_node("X3\n(Income)", color='lightblue')

G.add_node("H1", color='orange')
G.add_node("H2", color='orange')

G.add_node("Output\n(Loan Approval)", color='lightgreen')

# Add edges with weights as labels
G.add_edge("X1\n(Age)", "H1", weight=W1)
G.add_edge("X2\n(Debt Ratio)", "H1", weight=W2)
G.add_edge("X3\n(Income)", "H1", weight=W3)

G.add_edge("X1\n(Age)", "H2", weight=W4)
G.add_edge("X2\n(Debt Ratio)", "H2", weight=W5)
G.add_edge("X3\n(Income)", "H2", weight=W6)

G.add_edge("H1", "Output\n(Loan Approval)", weight=W7)
G.add_edge("H2", "Output\n(Loan Approval)", weight=W8)

# Positioning
pos = {
    "X1\n(Age)": (-2, 1),
    "X2\n(Debt Ratio)": (-2, 0),
    "X3\n(Income)": (-2, -1),

    "H1": (0, 0.7),
    "H2": (0, -0.7),

    "Output\n(Loan Approval)": (2, 0),
}

# Draw nodes
colors = [G.nodes[n]['color'] for n in G.nodes()]
nx.draw(G, pos, with_labels=True, node_color=colors, node_size=3000, font_size=9)

# Draw edge labels (weights)
edge_labels = {(u,v): f"{d['weight']}" for u,v,d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

plt.title("Loan Approval ANN (3 → 2 → 1)")
plt.tight_layout()
plt.savefig("loan_ann.png")
plt.show()

print("\nDiagram saved as: loan_ann.png")
