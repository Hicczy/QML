"""
Variational quantum classifier on real datasets — moons & iris
==============================================================

Where 02_variational_classifier.py (parity, 4-bit) and
02b_parity_generalization.py (parity, 5-bit with train/test split) used
discrete binary inputs and basis encoding, this script uses two real
small datasets with CONTINUOUS features. That forces a different
encoding choice — angle encoding — and a proper train/test split.

  Moons:    2D real-valued, 2 classes, non-linear decision boundary.
            Encoded into 2 qubits via R_Y(angle) per feature.
  Iris:     4D real-valued, 3 classes total. We use the binary subset
            {versicolor, virginica} — the harder pair, since setosa is
            linearly separable from everything.

For both we use the same architecture as the parity scripts (Rot+CNOT
layers, ⟨Z_0⟩ + bias readout, square-loss + Nesterov SGD), so the only
thing that changes between experiments is the data and the encoding.

Run:  python 02c_moons_iris.py
"""

import time
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.optimize import NesterovMomentumOptimizer
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# ----------------------------------------------------------------------
# Shared training infrastructure
# ----------------------------------------------------------------------

def make_circuit(n_qubits, n_layers):
    dev = qml.device("lightning.qubit", wires=n_qubits)

    def state_preparation(x):
        # Angle encoding: one R_Y per feature, scaled to [0, pi]
        for i in range(n_qubits):
            qml.RY(x[i], wires=i)

    def layer(weights):
        for w in range(n_qubits):
            qml.Rot(weights[w, 0], weights[w, 1], weights[w, 2], wires=w)
        for w in range(n_qubits):
            qml.CNOT(wires=[w, (w + 1) % n_qubits])

    @qml.qnode(dev, interface="autograd", diff_method="adjoint")
    def circuit(weights, x):
        state_preparation(x)
        for layer_weights in weights:
            layer(layer_weights)
        return qml.expval(qml.PauliZ(0))

    return circuit


def make_variational_classifier(n_qubits, n_layers):
    circuit = make_circuit(n_qubits, n_layers)

    def variational_classifier(weights, bias, x):
        return circuit(weights, x) + bias

    def square_loss(labels, predictions):
        return pnp.mean((labels - qml.math.stack(predictions)) ** 2)

    def accuracy(labels, predictions):
        preds = pnp.sign(qml.math.stack(predictions))
        return pnp.mean(preds == labels)

    def cost(weights, bias, X, Y):
        predictions = [variational_classifier(weights, bias, x) for x in X]
        return square_loss(Y, predictions)

    def eval_set(weights, bias, X, Y):
        preds = [variational_classifier(weights, bias, x) for x in X]
        return float(square_loss(Y, preds)), float(accuracy(Y, preds))

    return variational_classifier, cost, eval_set


def train(
    X_train, Y_train, X_test, Y_test,
    n_qubits, n_layers, n_iter, batch_size, step_size, seed, label
):
    _, cost, eval_set = make_variational_classifier(n_qubits, n_layers)

    pnp.random.seed(seed)
    weights = 0.05 * pnp.random.randn(n_layers, n_qubits, 3, requires_grad=True)
    bias    = pnp.array(0.0, requires_grad=True)
    opt     = NesterovMomentumOptimizer(stepsize=step_size)
    rng     = np.random.RandomState(seed)

    history = {"iter": [], "train_loss": [], "train_acc": [],
               "test_loss": [],  "test_acc": []}

    print(f"\n--- {label} ---")
    print(f"  qubits={n_qubits}  layers={n_layers}  "
          f"train={len(X_train)}  test={len(X_test)}")
    t0 = time.time()
    for it in range(n_iter):
        batch = rng.choice(len(X_train), batch_size, replace=False)
        weights, bias, _, _ = opt.step(
            cost, weights, bias, X_train[batch], Y_train[batch]
        )
        train_loss, train_acc = eval_set(weights, bias, X_train, Y_train)
        test_loss,  test_acc  = eval_set(weights, bias, X_test,  Y_test)
        history["iter"].append(it)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        if it % 5 == 0 or it == n_iter - 1:
            print(f"  iter {it:3d}  train_acc={train_acc:.3f}  "
                  f"test_acc={test_acc:.3f}  gap={train_acc - test_acc:+.3f}")
    print(f"  total time: {time.time() - t0:.1f}s")
    return history


# ----------------------------------------------------------------------
# Data prep
# ----------------------------------------------------------------------

def prep_moons(n_samples=200, noise=0.15, test_frac=0.5, seed=42):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    # Re-label to {-1, +1}
    y = np.where(y == 0, -1, 1)
    # Scale features to [0, pi] for angle encoding
    X = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(X)
    return train_test_split(X, y, test_size=test_frac, stratify=y, random_state=seed)


def prep_iris(test_frac=0.5, seed=42):
    """Versicolor (1) vs virginica (2) — the harder binary subset."""
    data = load_iris()
    mask = data.target != 0          # drop setosa (class 0)
    X = data.data[mask]              # 100 samples, 4 features
    y = data.target[mask]
    y = np.where(y == 1, -1, 1)      # versicolor → -1, virginica → +1
    X = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(X)
    return train_test_split(X, y, test_size=test_frac, stratify=y, random_state=seed)


# ----------------------------------------------------------------------
# Plot two histories side by side
# ----------------------------------------------------------------------

def plot_two(hist_moons, hist_iris, fname):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for col, (h, name) in enumerate([(hist_moons, "Moons"), (hist_iris, "Iris (versi vs virgi)")]):
        axes[0, col].plot(h["iter"], h["train_loss"], label="train")
        axes[0, col].plot(h["iter"], h["test_loss"],  label="test")
        axes[0, col].set_title(f"{name} — loss"); axes[0, col].legend()
        axes[0, col].set_xlabel("iter"); axes[0, col].set_ylabel("square loss")

        axes[1, col].plot(h["iter"], h["train_acc"], label="train")
        axes[1, col].plot(h["iter"], h["test_acc"],  label="test")
        axes[1, col].set_title(f"{name} — accuracy"); axes[1, col].legend()
        axes[1, col].set_xlabel("iter"); axes[1, col].set_ylabel("accuracy")
        axes[1, col].set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(fname, dpi=120)
    print(f"\nSaved {fname}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def verdict(history, name):
    train_acc = history["train_acc"][-1]
    test_acc = history["test_acc"][-1]
    gap = train_acc - test_acc
    print(f"\n{name}:  final train={train_acc:.3f}  test={test_acc:.3f}  gap={gap:+.3f}")
    if gap > 0.15:
        print("  -> OVERFIT")
    elif test_acc > 0.9:
        print("  -> GENERALIZED")
    elif train_acc < 0.7:
        print("  -> UNDERFIT (try more layers)")
    else:
        print("  -> PARTIAL")


def main():
    # ---------- Moons ----------
    Xtr, Xte, ytr, yte = prep_moons(n_samples=200, noise=0.15)
    Xtr = pnp.array(Xtr, requires_grad=False)
    ytr = pnp.array(ytr, requires_grad=False)
    Xte = pnp.array(Xte, requires_grad=False)
    yte = pnp.array(yte, requires_grad=False)
    h_moons = train(
        Xtr, ytr, Xte, yte,
        n_qubits=2, n_layers=4, n_iter=30,
        batch_size=10, step_size=0.3, seed=42, label="Moons (2 qubits, 4 layers)"
    )

    # ---------- Iris ----------
    Xtr, Xte, ytr, yte = prep_iris()
    Xtr = pnp.array(Xtr, requires_grad=False)
    ytr = pnp.array(ytr, requires_grad=False)
    Xte = pnp.array(Xte, requires_grad=False)
    yte = pnp.array(yte, requires_grad=False)
    h_iris = train(
        Xtr, ytr, Xte, yte,
        n_qubits=4, n_layers=3, n_iter=30,
        batch_size=8, step_size=0.3, seed=42, label="Iris versi-virgi (4 qubits, 3 layers)"
    )

    verdict(h_moons, "Moons")
    verdict(h_iris,  "Iris ")

    plot_two(h_moons, h_iris, "moons_iris_curves.png")


if __name__ == "__main__":
    main()
