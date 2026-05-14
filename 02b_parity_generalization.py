"""
Variational quantum classifier — parity, with a REAL train/test split
=====================================================================

Companion to 02_variational_classifier.py. That script's training curve
looked "too perfect" because we trained and tested on the same 16
examples — the model was just memorizing the truth table.

Here we use 6-bit parity (64 total examples) so we can hold out half for
testing. Now the model has to actually learn the parity STRUCTURE, not
memorize input-output pairs. The gap between train and test accuracy is
the honest story.

What to look for in the output:
  - If train accuracy goes to 1.0 but test accuracy stalls near 0.5,
    the model overfit / memorized. (Classic over-parametrization.)
  - If train and test both go to 1.0, the model truly learned the parity
    function and is generalizing.
  - If both stall low, the model lacks the capacity / right inductive
    bias to learn parity — try more layers.

Run:  python 02b_parity_generalization.py
"""

import numpy as np                # plain numpy for data prep
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.optimize import NesterovMomentumOptimizer
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
N_BITS       = 5                  # 2^5 = 32 examples total
N_LAYERS     = 3                  # parity at n=5 needs more depth than n=4
TRAIN_FRAC   = 0.5                # 16 train, 16 test
N_ITER       = 40
BATCH_SIZE   = 6
STEP_SIZE    = 0.3
SEED         = 7


# ----------------------------------------------------------------------
# Quantum device & ansatz — same shape as 02_variational_classifier.py,
# just parametrized by n_qubits.
# ----------------------------------------------------------------------
n_qubits = N_BITS
dev = qml.device("default.qubit", wires=n_qubits)


def state_preparation(x):
    qml.BasisState(x, wires=range(n_qubits))


def layer(weights):
    for wire in range(n_qubits):
        qml.Rot(weights[wire, 0], weights[wire, 1], weights[wire, 2], wires=wire)
    for wire in range(n_qubits):
        qml.CNOT(wires=[wire, (wire + 1) % n_qubits])


@qml.qnode(dev, interface="autograd")
def circuit(weights, x):
    state_preparation(x)
    for layer_weights in weights:
        layer(layer_weights)
    return qml.expval(qml.PauliZ(0))


def variational_classifier(weights, bias, x):
    return circuit(weights, x) + bias


# ----------------------------------------------------------------------
# Loss / accuracy
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# Data: 6-bit parity, split into train/test
# ----------------------------------------------------------------------
def build_parity_dataset(n_bits):
    X = np.array(
        [[int(b) for b in format(i, f"0{n_bits}b")] for i in range(2 ** n_bits)],
        dtype=int,
    )
    Y = np.array([1 if sum(x) % 2 == 0 else -1 for x in X])
    return X, Y


def main():
    rng = np.random.RandomState(SEED)
    X_all, Y_all = build_parity_dataset(N_BITS)
    print(f"Dataset: {len(X_all)} examples, {N_BITS}-bit parity")
    print(f"  class balance: +1={int((Y_all == 1).sum())}, "
          f"-1={int((Y_all == -1).sum())}")

    # Stratified shuffle: half the +1s and half the -1s in train
    pos = np.where(Y_all == 1)[0]
    neg = np.where(Y_all == -1)[0]
    rng.shuffle(pos); rng.shuffle(neg)
    n_pos_train = int(len(pos) * TRAIN_FRAC)
    n_neg_train = int(len(neg) * TRAIN_FRAC)
    train_idx = np.concatenate([pos[:n_pos_train], neg[:n_neg_train]])
    test_idx  = np.concatenate([pos[n_pos_train:], neg[n_neg_train:]])
    rng.shuffle(train_idx); rng.shuffle(test_idx)

    # Wrap in PennyLane numpy so they flow into autograd cleanly
    X_train = pnp.array(X_all[train_idx], requires_grad=False)
    Y_train = pnp.array(Y_all[train_idx], requires_grad=False)
    X_test  = pnp.array(X_all[test_idx],  requires_grad=False)
    Y_test  = pnp.array(Y_all[test_idx],  requires_grad=False)
    print(f"  train: {len(X_train)}  |  test: {len(X_test)}")

    # Init params
    pnp.random.seed(SEED)
    weights = 0.01 * pnp.random.randn(N_LAYERS, n_qubits, 3, requires_grad=True)
    bias    = pnp.array(0.0, requires_grad=True)

    opt = NesterovMomentumOptimizer(stepsize=STEP_SIZE)

    history = {"iter": [], "train_loss": [], "train_acc": [],
               "test_loss": [],  "test_acc": []}

    print("\nTraining ...")
    for it in range(N_ITER):
        # Mini-batch SGD on TRAIN set only
        batch_idx = rng.choice(len(X_train), BATCH_SIZE, replace=False)
        X_batch = X_train[batch_idx]
        Y_batch = Y_train[batch_idx]
        weights, bias, _, _ = opt.step(cost, weights, bias, X_batch, Y_batch)

        # Evaluate on both sets every iteration
        train_loss, train_acc = eval_set(weights, bias, X_train, Y_train)
        test_loss,  test_acc  = eval_set(weights, bias, X_test,  Y_test)

        history["iter"].append(it)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        if it % 5 == 0 or it == N_ITER - 1:
            gap = train_acc - test_acc
            print(f"  iter {it:3d}  "
                  f"train: loss={train_loss:.3f} acc={train_acc:.3f}  "
                  f"test: loss={test_loss:.3f} acc={test_acc:.3f}  "
                  f"gap={gap:+.3f}")

    # ------- Final inspection -------
    print(f"\nFinal train acc: {history['train_acc'][-1]:.3f}")
    print(f"Final test  acc: {history['test_acc'][-1]:.3f}")
    gap = history["train_acc"][-1] - history["test_acc"][-1]
    if gap > 0.15:
        verdict = "OVERFIT: model memorized training set, didn't learn parity"
    elif history["test_acc"][-1] > 0.9:
        verdict = "GENERALIZED: model truly learned the parity structure"
    elif history["train_acc"][-1] < 0.8:
        verdict = "UNDERFIT: not enough capacity — try more layers"
    else:
        verdict = "PARTIAL: model is somewhere in between"
    print(f"Verdict: {verdict}")

    # ------- Plot -------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["iter"], history["train_loss"], label="train")
    axes[0].plot(history["iter"], history["test_loss"],  label="test")
    axes[0].set_xlabel("iteration"); axes[0].set_ylabel("square loss")
    axes[0].set_title("Loss"); axes[0].legend()

    axes[1].plot(history["iter"], history["train_acc"], label="train")
    axes[1].plot(history["iter"], history["test_acc"],  label="test")
    axes[1].set_xlabel("iteration"); axes[1].set_ylabel("accuracy")
    axes[1].set_title("Accuracy"); axes[1].set_ylim(-0.05, 1.05); axes[1].legend()

    plt.tight_layout()
    plt.savefig("parity_generalization_curves.png", dpi=120)
    print("\nSaved parity_generalization_curves.png")
    plt.show()


if __name__ == "__main__":
    main()
