# A bridge from classical ML to QML

You already know how supervised learning works in the classical world: pick a model family $f_\theta$, define a loss $L(\theta)$ on labeled data, and use gradient descent to find good parameters $\theta$. **Variational quantum machine learning (QML) is the same recipe** — the only thing that changes is what $f_\theta$ is. Instead of a linear function, a tree, or a neural net, $f_\theta$ is now a *parametrized quantum circuit* whose final measurement gives you a number you treat as a prediction.

This primer gets you from "CS student who has done logistic regression and a basic NN" to "I can read the PennyLane variational classifier tutorial without getting stuck on quantum jargon." It is intentionally not rigorous about the physics — that's fine for now. You can fill in rigor later.

---

## 1. The minimum quantum vocabulary you need

### Qubit
A single qubit's state is a unit vector in $\mathbb{C}^2$:
$$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle, \quad |\alpha|^2 + |\beta|^2 = 1.$$
The two basis vectors $|0\rangle = \begin{pmatrix}1\\0\end{pmatrix}$ and $|1\rangle = \begin{pmatrix}0\\1\end{pmatrix}$ are like the "labels" 0 and 1, but the qubit can be in a *superposition* — a complex linear combination — until you measure it.

When you measure in the computational basis, you get outcome 0 with probability $|\alpha|^2$ and outcome 1 with probability $|\beta|^2$. So a qubit's state encodes a probability distribution over $\{0, 1\}$, plus extra phase information.

### $n$ qubits
The state of $n$ qubits is a unit vector in $\mathbb{C}^{2^n}$. That exponential is *the* reason quantum computing is interesting: simulating a 30-qubit system on a laptop already needs gigabytes of RAM. A real quantum computer represents that vector "for free" in its hardware.

### Gates
Operations on qubits are *unitary matrices* (linear, length-preserving, reversible). A gate acting on $k$ qubits is a $2^k \times 2^k$ unitary. The familiar single-qubit gates:

- **Pauli X** = bit-flip: $\begin{pmatrix}0&1\\1&0\end{pmatrix}$ (swaps $|0\rangle \leftrightarrow |1\rangle$)
- **Hadamard** = creates superposition: takes $|0\rangle \to \frac{1}{\sqrt2}(|0\rangle + |1\rangle)$
- **Rotations** $R_X(\theta), R_Y(\theta), R_Z(\theta)$ = continuous, parametrized single-qubit gates. **These are where your trainable parameters live.**

Two-qubit entangling gate you'll see most:
- **CNOT**: flip the target qubit if the control qubit is $|1\rangle$. This is what creates correlations between qubits — without it, your circuit is just $n$ independent single-qubit problems.

### Circuit
A *quantum circuit* is just a sequence of gates applied to a register of qubits — read left to right, like a forward pass through a neural network. You start in $|0\dots0\rangle$, apply gates, and at the end measure something.

### Measurement (the readout)
After the circuit, you measure an *observable* — usually $\langle Z_0 \rangle$, the expected value of the Pauli-Z operator on qubit 0. This is a real number in $[-1, +1]$. **This number is what you treat as your model output.** It plays the same role as the logit/score in logistic regression.

---

## 2. The 4-step variational recipe

This is the entire architecture of a variational classifier. Memorize it:

| Step | Classical analog | Quantum version |
|------|------------------|------------------|
| 1. Encode input $x$ into the model | First layer of your NN takes $x$ as input | A *feature-map circuit* $S(x)$ prepares a state $\|\phi(x)\rangle = S(x)\|0\dots0\rangle$ |
| 2. Apply the trainable model | Hidden layers $f_\theta$ | A *variational ansatz* $U(\theta)$ — a circuit of rotation gates parametrized by $\theta$, possibly with entangling layers |
| 3. Read out a prediction | Output layer + sigmoid/softmax | Measure an observable, e.g. $\langle Z_0 \rangle$, giving a number in $[-1, +1]$ |
| 4. Optimize | SGD on a loss | Same: compute a loss between predictions and labels, take gradients with respect to $\theta$, update |

So your model is:
$$f_\theta(x) = \langle 0\dots0 | S^\dagger(x) U^\dagger(\theta)\, Z_0\, U(\theta) S(x) | 0\dots0\rangle$$

That intimidating expression is just "encode $x$, apply the trainable circuit, measure $Z_0$." It's a function $\mathbb{R}^d \to [-1, +1]$ that you can plug into any classical loss function.

### Where the parameters live
$\theta$ is just a NumPy array. In PennyLane, you literally write `weights = np.random.randn(num_layers, num_qubits, 3)` and pass it into the QNode. The "3" is because each `qml.Rot` gate has three angles (it's a general single-qubit rotation).

### How gradients work
Two big options:
1. **Simulation autodiff**: when you simulate the circuit on a classical computer, you can backprop through the matrix operations like any other tensor program. PennyLane integrates with autograd / PyTorch / JAX / TF.
2. **Parameter-shift rule**: a beautiful trick where the analytic gradient of $\langle Z \rangle$ with respect to a rotation angle $\theta$ equals
$$\frac{\partial \langle Z\rangle}{\partial \theta} = \frac{1}{2}\bigl(\langle Z\rangle_{\theta + \pi/2} - \langle Z\rangle_{\theta - \pi/2}\bigr).$$
This is exact and works even on real quantum hardware, where you can't backprop. It's the QML equivalent of finite differences but mathematically exact.

You don't have to think about which one is being used — PennyLane handles it. But it's good to know the parameter-shift rule exists, because it's *the* reason variational QML is even possible on near-term hardware.

---

## 3. What the PennyLane tutorial is doing

The tutorial has two examples:

### Example 1: Parity function (4 input bits → ±1 label)
- **Data**: all 16 strings $x \in \{0,1\}^4$. Label is $+1$ if the number of 1s is even, $-1$ if odd.
- **Encoding $S(x)$**: `qml.BasisState(x, wires=...)` — literally prepare $|x\rangle$. The trivial encoding.
- **Ansatz $U(\theta)$**: a few "layers," each consisting of a `Rot` on every qubit followed by a ring of CNOTs. The `Rot` provides the trainable knobs; the CNOTs entangle so the model isn't trivially separable.
- **Readout**: $\langle Z_0\rangle$, plus a learned bias. Predict $\mathrm{sign}(f_\theta(x) + b)$.
- **Loss**: square loss against $\pm 1$ labels.
- **Optimizer**: Nesterov momentum (just SGD with momentum).

This task is the QML "Hello, World." Parity is famously the function classical perceptrons can't learn — though deep nets can — and the variational classifier learns it cleanly with a few qubits. It's a *demonstration* example, not a benchmark of quantum advantage.

### Example 2: Iris (continuous features → class label)
Same recipe, different encoding: the inputs are real-valued, so `BasisState` no longer works. The tutorial uses *amplitude encoding* (cram the feature vector into the amplitudes of a quantum state, after normalization and padding). Everything else is the same.

The Iris example exists to show that **the choice of encoding $S(x)$ is the critical design decision in QML**, more than the ansatz. Different encodings produce wildly different inductive biases and, in some theoretical analyses, determine whether the model can outperform classical baselines at all.

---

## 4. The honest landscape (relevant for your project)

QML is a young field. Some categories you'll see in the literature:

- **Variational quantum classifiers / QNNs** (what the tutorial covers). Train a parametrized circuit end-to-end. Flexible but suffers from *barren plateaus* — gradients vanish exponentially in qubit count for generic ansätze. This is the main open problem.
- **Quantum kernel methods**. Use a quantum circuit only to compute a kernel $k(x, x') = |\langle\phi(x)|\phi(x')\rangle|^2$, then plug into a classical SVM. Cleaner theoretical story, no barren plateaus.
- **VQE / QAOA**. Variational algorithms for chemistry / combinatorial optimization. Not really ML, but use the same circuit-as-model trick.
- **Quantum-inspired classical algorithms**. Ewin Tang's line of work showing some "quantum advantages" can be replicated classically. Important context.

**A semester project is enough time to do one of these well**, not to invent something new. Realistic project shapes:
1. *Empirical*: pick a small dataset, compare a variational classifier vs. a kernel method vs. a classical baseline. Study how performance scales with qubits, layers, encoding choice. Write up what you find.
2. *Methods comparison*: implement two encoding schemes (amplitude vs. angle vs. data re-uploading) on the same task and analyze.
3. *Trainability*: empirically study barren plateaus — measure gradient variance vs. number of qubits/layers. Reproduce a published result.
4. *Reproduction*: pick a concrete QML paper (there are good ones on quantum kernels for HEP data, or on data re-uploading classifiers) and reproduce its main figure.

Once we've worked through the parity classifier and you have your hands on PennyLane, we can pick a direction.

---

## 5. Reading guide for the tutorial

Now, when you read the PennyLane page, here's the map:

- The `layer(W)` function is your **trainable layer** — one block of the ansatz.
- `statepreparation(x)` is your **feature map** $S(x)$.
- `circuit(weights, x)` is the full QNode: state-prep, ansatz, measurement.
- `variational_classifier(weights, bias, x)` adds a learnable scalar bias to the circuit's expectation value — same role as the bias term in logistic regression.
- `square_loss`, `accuracy`, `cost` are exactly what they look like.
- The training loop is vanilla SGD with mini-batches.

If anything in the tutorial trips you up, the answer is almost always in the table in section 2 above.

Open `02_variational_classifier.py` next — it implements the parity example end-to-end with comments tied back to this primer.
