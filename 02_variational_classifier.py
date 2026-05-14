"""
Variational quantum classifier — parity function
================================================

Companion script to 01_qml_concepts.md. Implements the same parity example
from the PennyLane tutorial, with comments tied back to the four-step recipe:

    Step 1: encode x  -->  state_preparation(x)
    Step 2: trainable circuit U(theta)  -->  layer(weights) repeated
    Step 3: measurement readout  -->  qml.expval(qml.PauliZ(0))
    Step 4: optimize a loss  -->  classical SGD on the cost function

Run it as a script:    python 02_variational_classifier.py
Or step through cells in VS Code / Jupyter using the `# %%` markers.

Setup (one time):
    pip install pennylane matplotlib
"""

# %% [markdown]
# ## Imports
#
# We use PennyLane's wrapped NumPy (`pennylane.numpy`). It's a thin layer on
# top of autograd that lets PennyLane compute gradients of circuit parameters
# automatically. For our purposes it behaves exactly like normal NumPy.

# %%
import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer
import matplotlib.pyplot as plt

# Reproducibility
np.random.seed(0)


# %% [markdown]
# ## 1. The quantum device
#
# PennyLane separates the *abstract circuit* from the *backend* that runs it.
# `default.qubit` is the built-in state-vector simulator: exact, fast for
# small systems, runs on your laptop. To run on real hardware you'd swap this
# out for an IBM/IonQ/etc. device — the rest of the code wouldn't change.

# %%
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)


# %% [markdown]
# ## 2. State preparation — the feature map S(x)
#
# Step 1 of the recipe. We're encoding a 4-bit input x in {0,1}^4 as the
# computational basis state |x>. `BasisState` is the simplest possible
# encoding: literally "set qubit i to |x_i>". For real-valued inputs you'd
# use angle or amplitude encoding instead — see the Iris example in the
# PennyLane tutorial.

# %%
def state_preparation(x):
    qml.BasisState(x, wires=range(n_qubits))


# %% [markdown]
# ## 3. The variational ansatz — one trainable layer U_l(theta_l)
#
# Step 2. Each layer has two parts:
#
#   (a) A general single-qubit rotation `Rot(phi, theta, omega)` on every
#       qubit. This gives every qubit three trainable angles. `Rot` is just
#       Rz * Ry * Rz under the hood — an arbitrary single-qubit unitary.
#
#   (b) A ring of CNOTs that entangles neighboring qubits. Without this,
#       each qubit evolves independently and the model can't represent any
#       function that depends on more than one bit at a time. Parity
#       depends on ALL the bits, so entanglement is essential.

# %%
def layer(weights):
    # weights has shape (n_qubits, 3): three rotation angles per qubit
    for wire in range(n_qubits):
        qml.Rot(weights[wire, 0], weights[wire, 1], weights[wire, 2], wires=wire)
    # CNOT ring: 0->1, 1->2, 2->3, 3->0
    for wire in range(n_qubits):
        qml.CNOT(wires=[wire, (wire + 1) % n_qubits])


# %% [markdown]
# ## 4. The full QNode — encoding + ansatz + measurement
#
# `@qml.qnode(dev)` turns the function below into a *quantum node*: a callable
# that, when invoked, builds and runs the circuit on `dev` and returns the
# measurement result. From the outside it just looks like a NumPy function:
# inputs in, scalar out, differentiable.
#
# We measure <Z_0>, the expected value of Pauli-Z on qubit 0. This is a
# real number in [-1, +1] — exactly the right shape for a binary classifier.

# %%
@qml.qnode(dev, interface="autograd")
def circuit(weights, x):
    state_preparation(x)            # Step 1: encode x
    for layer_weights in weights:   # Step 2: stack of trainable layers
        layer(layer_weights)
    return qml.expval(qml.PauliZ(0))  # Step 3: readout


# %% [markdown]
# ## 5. Wrap as a classifier with a bias term
#
# Adding a learnable scalar bias plays the exact role it does in logistic
# regression: it shifts the decision boundary. Without it, the model is
# constrained to predict 0 when the encoded state happens to give <Z_0> = 0.

# %%
def variational_classifier(weights, bias, x):
    return circuit(weights, x) + bias


# %% [markdown]
# ## 6. Loss and accuracy
#
# Square loss against ±1 labels. This is a regression-style loss applied to
# a classification problem — fine for binary tasks where labels are {-1, +1}
# and the model output is in [-1, +1].

# %%
def square_loss(labels, predictions):
    return np.mean((labels - qml.math.stack(predictions)) ** 2)

def accuracy(labels, predictions):
    preds = np.sign(qml.math.stack(predictions))
    return np.mean(preds == labels)

def cost(weights, bias, X, Y):
    predictions = [variational_classifier(weights, bias, x) for x in X]
    return square_loss(Y, predictions)


# %% [markdown]
# ## 7. The dataset — 4-bit parity
#
# All 16 binary strings of length 4. Label is +1 if the number of 1s is even
# (parity 0), else -1. This is famous as a function a single-layer perceptron
# cannot learn — it's the canonical "you need nonlinearity" example. A small
# variational circuit handles it without breaking a sweat.

# %%
X = np.array([[int(b) for b in format(i, "04b")] for i in range(16)], requires_grad=False)
Y = np.array([1 if sum(x) % 2 == 0 else -1 for x in X], requires_grad=False)

print("Dataset:")
for x, y in zip(X, Y):
    print(f"  {x}  ->  {y:+d}")


# %% [markdown]
# ## 8. Initialize parameters
#
# `requires_grad=True` flags these arrays as the things to differentiate
# with respect to. PennyLane's autograd-backed numpy uses this to build
# the gradient computation graph.

# %%
num_layers = 2
weights_init = 0.01 * np.random.randn(num_layers, n_qubits, 3, requires_grad=True)
bias_init = np.array(0.0, requires_grad=True)


# %% [markdown]
# ## 9. Training loop
#
# Vanilla mini-batch SGD with Nesterov momentum. Each iteration picks a
# random subset of size `batch_size`, computes the gradient of `cost` with
# respect to (weights, bias), and updates.

# %%
opt = NesterovMomentumOptimizer(stepsize=0.5)
batch_size = 5
num_iterations = 30

weights = weights_init
bias = bias_init

history = {"iter": [], "loss": [], "acc": []}

for it in range(num_iterations):
    # Sample a mini-batch
    batch_idx = np.random.randint(0, len(X), (batch_size,))
    X_batch = X[batch_idx]
    Y_batch = Y[batch_idx]

    # Gradient step: opt.step takes the cost and the args to differentiate
    weights, bias, _, _ = opt.step(cost, weights, bias, X_batch, Y_batch)

    # Track full-dataset metrics each iteration so we can plot
    predictions = [variational_classifier(weights, bias, x) for x in X]
    current_loss = square_loss(Y, predictions)
    current_acc = accuracy(Y, predictions)
    history["iter"].append(it)
    history["loss"].append(float(current_loss))
    history["acc"].append(float(current_acc))

    if it % 2 == 0 or it == num_iterations - 1:
        print(f"Iter {it:3d}  loss={current_loss:.4f}  acc={current_acc:.3f}")


# %% [markdown]
# ## 10. Inspect the trained model
#
# Print the predictions on the full dataset alongside the labels. Every row
# should match if training succeeded.

# %%
print("\nFinal predictions:")
print(f"{'x':<14} {'true':>5} {'<Z_0>+b':>12} {'pred':>6}")
for x, y in zip(X, Y):
    raw = float(variational_classifier(weights, bias, x))
    pred = int(np.sign(raw))
    mark = "OK" if pred == y else "X"
    x_str = "".join(str(int(b)) for b in x)
    print(f"{x_str:<14} {int(y):+5d} {raw:+12.4f} {pred:+6d}  {mark}")


# %% [markdown]
# ## 11. Plots
#
# Two subplots: training loss and full-dataset accuracy across iterations.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(history["iter"], history["loss"])
axes[0].set_xlabel("iteration"); axes[0].set_ylabel("square loss"); axes[0].set_title("Loss")
axes[1].plot(history["iter"], history["acc"])
axes[1].set_xlabel("iteration"); axes[1].set_ylabel("accuracy"); axes[1].set_title("Accuracy on full dataset")
axes[1].set_ylim(-0.05, 1.05)
plt.tight_layout()
plt.savefig("training_curves.png", dpi=120)
print("\nSaved training_curves.png")
plt.show()


# %% [markdown]
# ## 12. Things to try (project warm-ups)
#
# Once the above runs cleanly, these are short, instructive experiments:
#
# 1. **Vary depth.** Set `num_layers` to 1, 2, 4, 8. Does deeper always help?
#    (Spoiler: not for this task — and for harder tasks, very deep ansätze
#    hit barren plateaus where gradients vanish.)
#
# 2. **Change the entangling pattern.** Try a linear chain of CNOTs instead
#    of a ring, or only every-other-qubit entanglement. How does
#    expressivity change?
#
# 3. **Inspect the parameter-shift rule directly.** Pick one weight, compute
#    the gradient via PennyLane, then verify it matches
#    (cost(theta + pi/2) - cost(theta - pi/2)) / 2 manually. This is the
#    rule that makes QML on real hardware possible.
#
# 4. **Swap the optimizer.** Try `qml.AdamOptimizer(0.1)` or
#    `qml.GradientDescentOptimizer(0.5)`. Same convergence story as classical.
#
# 5. **Try a different feature map.** Replace `BasisState` with
#    `qml.AngleEmbedding(x, wires=range(n_qubits))`. You'll need to convert
#    the {0,1} bits to angles (e.g., 0 -> 0, 1 -> pi). This is the bridge
#    to handling continuous data.
#
# 6. **Move to the Iris dataset.** That's the second half of the tutorial.
#    The encoding gets more interesting (amplitude embedding) and you start
#    seeing that ENCODING IS THE KEY DESIGN DECISION in variational QML.
