"""Train the from-scratch MLP on the two-moons toy dataset and plot the
learned decision boundary, to sanity-check that manual backprop works."""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons

from mlp import MLP, binary_cross_entropy

EPOCHS = 8000
LR = 1.0
PRINT_EVERY = 1000


def main():
    X, y = make_moons(n_samples=500, noise=0.2, random_state=0)
    y = y.reshape(-1, 1).astype(np.float64)

    # Standardize features so gradient descent converges cleanly.
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    model = MLP(layer_sizes=[2, 16, 16, 1], activations=["relu", "relu", "sigmoid"], seed=0)

    for epoch in range(1, EPOCHS + 1):
        loss = model.train_step(X, y, lr=LR)
        if epoch % PRINT_EVERY == 0 or epoch == 1:
            preds = (model.predict(X) > 0.5).astype(np.float64)
            acc = (preds == y).mean()
            print(f"epoch {epoch:5d}  loss={loss:.4f}  acc={acc:.3f}")

    final_preds = (model.predict(X) > 0.5).astype(np.float64)
    final_acc = (final_preds == y).mean()
    print(f"\nFinal training accuracy: {final_acc:.3f}")

    plot_decision_boundary(model, X, y)


def plot_decision_boundary(model, X, y, out_path="decision_boundary.png"):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, probs, levels=50, cmap="RdBu", alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap="RdBu", edgecolors="k", s=20)
    plt.title("From-scratch MLP decision boundary (two moons)")
    plt.savefig(out_path, dpi=150)
    print(f"Saved decision boundary plot to {out_path}")


if __name__ == "__main__":
    main()
