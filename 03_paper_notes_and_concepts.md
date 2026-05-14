# Study notes: papers + three conceptual deep-dives

Companion to `01_qml_concepts.md`. Covers two assigned readings and the three concepts your instructor flagged (dataset asymmetry, encodings + rotations, bias).

---

## Paper 1: QuKAN (Kiefer-Emmanouilidis et al., *Scientific Reports*, Oct 2025)

**Full title:** *QuKAN: A Quantum Circuit Born Machine Approach to Quantum Kolmogorov Arnold Networks*
**DOI:** 10.1038/s41598-025-22705-9

### The one-paragraph version
KAN (Kolmogorov-Arnold Networks, Liu et al. 2024) is a recent classical architecture that puts **learnable univariate functions on the edges** of a network — typically B-splines — instead of fixed scalar weights with fixed nonlinearities on nodes (like MLPs). KAN's appeal is interpretability and parameter efficiency. The QuKAN paper proposes a **quantum** version: they encode the B-spline basis functions into the amplitudes of a quantum state using a *Quantum Circuit Born Machine* (QCBM), then train a parametrized circuit on a "labelling" qubit register to learn the spline weights. Two flavours: a **hybrid** QuKAN (classical KAN scaffolding with quantum residual functions) and a **fully quantum** QuKAN. They test on moons, Iris, and two-variable function regression, and compare to vanilla variational quantum classifiers (VQCs) with amplitude and angle encodings.

### Key concepts the paper uses (and why they matter for you)

**Quantum Circuit Born Machine (QCBM).** A generative-quantum-model paradigm: prepare $|0\rangle^{\otimes n}$, apply a parametrized circuit $U(\theta)$, measure in the computational basis. The output is a probability distribution over basis states via the Born rule $p_\theta(x) = |\langle x | U(\theta) | 0\rangle|^2$. You train $\theta$ so this distribution matches a target distribution (here, via Maximum Mean Discrepancy loss).

**Label / position register split.** They split the qubits into two groups: *labelling qubits* indexing which basis function (which spline), and *position qubits* indexing the input $x$. The full state is $|\Psi\rangle = \sum_i c_i |i\rangle_{\text{label}} \otimes |\psi_i\rangle_{\text{pos}}$. This is the trick that lets one quantum state encode *multiple* functions simultaneously in superposition.

**Strongly entangling layers.** The ansatz they use is from Schuld et al. 2020: each layer has parametrized rotations on every qubit followed by nearest-neighbour CNOTs — same architecture pattern as our parity script in `02_variational_classifier.py`, just generalized.

### Results worth knowing
- **Moons (1000 train / 1000 test, noise=0.1):** hybrid QuKAN outperforms VQCs with both amplitude embedding and angle embedding (including ZZ feature map). It also beats Ivashkov et al.'s QKAN. Decision boundaries look similar to classical pyKAN but slightly less smooth.
- **Iris:** QuKAN competitive with classical pyKAN and ahead of the quantum baselines.
- **Function regression:** works on $f(x_1, x_2) = 2x_1 - 3x_2 + 1$ and $f(x_0, x_1) = \ln(x_0/x_1)$; matches or beats Wakaura et al.'s EVQKAN.
- **Pre-training ablation:** without the QCBM-encoded splines (replaced by Hadamard-uniform initialization), training plateaus at ~87.6%. So the spline pre-training is doing real work, not just decoration.

### Honest framing
This is a **methodology paper**, not a "quantum beats classical" paper. The benchmarks are deliberately toy. The win is architectural: showing that KAN-style learnable basis functions can live inside a quantum model, with a clean interpretability story. For your project, this is one of the better-documented "what does a real QML model look like" papers.

---

## Paper 2: arxiv 2604.07639 — *Exponential quantum advantage in processing massive classical data*

**Status:** title only — I could not retrieve the body. The PDF returned by the fetcher came back empty.

**Two possibilities to resolve this:**
1. **Recommended:** download the PDF from arxiv and drop it in `C:\Users\ayako\OneDrive\Documents\Claude\Projects\QML\` (or paste the abstract into chat). With the PDF in your folder I can read it directly.
2. Ask your workspace admin to add `arxiv.org` to the network allowlist.

**What I can infer just from the title** (and will overwrite once I see the actual paper): the title language — "exponential quantum advantage" + "massive classical data" — typically signals work on either (a) quantum-enhanced data structures like QRAM, (b) HHL-style linear-system algorithms applied to ML, or (c) random-feature / kernel methods that achieve provable separation for specific problems. The phrase is heavy — strong provable-advantage claims for *classical* data are rare and almost always come with strong caveats about data-loading cost (which is itself the famous bottleneck for "quantum-on-classical-data" advantage claims). Don't take the title at face value before reading; check exactly what model of computation they assume and what state-prep / oracle access they require. That's where the asterisks usually live.

---

## Concept 1 — Why QML datasets differ from classical ML datasets

> Your instructor's framing: moons is doable in QML but MNIST isn't, even though classical ML eats MNIST for breakfast.

Three forces are at work.

### A. The encoding bottleneck
Classical data has to be loaded into a quantum state before any quantum processing can happen. There is no free lunch — the loading itself is a circuit:

- **Angle encoding** uses $n$ qubits for $n$ features. MNIST has 784 features ⇒ 784 qubits. Largest current hardware: ~1000 noisy qubits, but you also need qubits for the ansatz and ancillas. Realistically infeasible.
- **Amplitude encoding** uses $\lceil \log_2(d) \rceil$ qubits for a $d$-dim feature vector — MNIST fits in 10 qubits. But state preparation requires **$O(2^n)$ gates** in the worst case (Möttönen et al. 2005, cited in the QuKAN paper). That's a deep circuit you have to run *every forward pass*, every gradient step, every test point. The QuKAN paper itself calls out this exponential depth scaling as the reason they avoid pure amplitude embedding.
- **Basis encoding** is essentially "one bit per qubit" — fine for binary features, hopeless for high-dim real-valued data.

Moons has 2 features. Iris has 4. Wine, Breast Cancer, parity — same regime. These are the datasets where encoding is cheap and the rest of the circuit dominates the discussion. **Encoding cost is the single most important practical reason QML lives in the "small dataset" world.**

### B. NISQ hardware reality (Noisy Intermediate-Scale Quantum)
Even when you can encode a dataset, today's hardware has:
- Limited qubit counts (~100–1000 qubits, decreasing rapidly with quality requirements).
- Short coherence times: each gate degrades the state slightly. Deep circuits accumulate error fast.
- No error correction yet at the scale ML needs.

So even *if* you amplitude-encode MNIST in 10 qubits, the state-prep depth alone uses up your error budget before any learning circuit runs.

### C. Trainability — barren plateaus
This is the QML-specific failure mode. McClean et al. (2018) showed that for sufficiently expressive parametrized circuits on $n$ qubits, the variance of the loss gradient with respect to any parameter scales as $\text{Var}[\partial_\theta L] \sim 1/2^n$. Translation: gradients vanish exponentially in qubit count for generic ansätze. Your optimizer sees flat noise instead of a landscape.

This means *you can't just scale up qubits to attack bigger datasets* — naive scaling kills training. Mitigation strategies (problem-specific ansätze, equivariant circuits, layer-wise training) are active research. Open question.

### D. Inductive biases
A CNN on MNIST works because *convolutional structure matches image structure*: translation invariance, local correlation. A generic variational circuit has none of those biases built in. So even if you could fit MNIST through a circuit, you'd be asking the optimizer to learn the equivalent of CNN structure from scratch, in a noisy non-convex landscape, with vanishing gradients. The result is that even tiny MNIST works far worse than a 1990s SVM.

### What this means for your project
QML datasets are small, low-dim, and often synthetic for a reason. The interesting comparisons aren't "QML vs. classical on MNIST" — those are publicity stunts. The interesting questions are:
- On small structured data (chemistry, finance time series, graph problems), can a *well-designed* QML model match classical models with fewer parameters?
- Are there problems with **quantum-native data** (states from a sensor, quantum circuit outputs) where the encoding is free and quantum has a real shot?
- Can quantum kernel methods (which avoid barren plateaus) carve out a niche?

If you want a concrete project shape, the QuKAN paper is a template: pick 1–2 toy datasets (moons + Iris), implement 2–3 model variants, study where each wins and why.

---

## Concept 2 — Amplitude, angle, basis encoding + how rotations work

### The three encodings side-by-side

| | **Basis** | **Angle** | **Amplitude** |
|---|---|---|---|
| Input shape | binary vector, length $n$ | real vector, length $n$ | real vector, length $d$ |
| Qubits needed | $n$ | $n$ | $\lceil \log_2 d \rceil$ |
| State prep depth | $O(n)$ (just X gates) | $O(n)$ (one rotation per qubit) | $O(d)$ worst case (Möttönen) |
| Expressivity | one bit per qubit — wasteful | low (kernel = trig functions) | high — every amplitude is a feature |
| Best when | binary inputs, you have plenty of qubits | $n$ small, want shallow circuits | high-dim data + you can afford deep state prep |
| Trained yet? | no — encoding is fixed once $x$ is chosen | depends on variant | yes if you use trainable feature maps |

#### Basis encoding — what `02_variational_classifier.py` uses
Given $x = (x_1, \dots, x_n) \in \{0,1\}^n$, prepare $|x_1 x_2 \dots x_n\rangle$. PennyLane's `qml.BasisState(x, wires=range(n))` does this in one call. Implementation: apply $X$ to qubit $i$ iff $x_i = 1$. Trivial cost, zero trainable parameters.

#### Angle encoding
For each feature $x_i \in \mathbb{R}$ (typically rescaled to $[0, \pi]$ or $[-\pi, \pi]$):
$$\text{Apply } R_Y(x_i) \text{ to qubit } i, \quad |0\rangle \to \cos(x_i/2)|0\rangle + \sin(x_i/2)|1\rangle.$$
The full encoded state is a tensor product of $n$ rotated qubits. The implicit kernel induced by this encoding turns out to be a product of $\cos^2$ and $\sin^2$ terms — basically a trigonometric Fourier-like feature map.

**Variants you'll see:**
- `qml.AngleEmbedding(x, wires=...)`: PennyLane's basic angle encoder.
- **Dense angle encoding:** pack two features per qubit by using both $R_Y$ and $R_Z$.
- **Data re-uploading** (Pérez-Salinas et al. 2020): interleave the encoding rotations with trainable layers — *encode, train, encode, train, ...*. This is one of the most expressive shallow approaches and avoids the "single-shot encoding limits you forever" problem.
- **ZZ feature map** (the QuKAN paper uses this as a baseline): angle encoding followed by entangling $R_{ZZ}(x_i x_j)$ gates — adds pairwise data correlations into the encoding itself. Roughly the quantum analog of polynomial features.

#### Amplitude encoding
For $x \in \mathbb{R}^d$ with $d = 2^n$ and $\|x\| = 1$:
$$|\psi(x)\rangle = \sum_{i=0}^{d-1} x_i |i\rangle.$$
Every amplitude is a feature. Exponentially compact in qubit count. But:
- **Cost:** Möttönen's algorithm for an arbitrary $|\psi(x)\rangle$ uses $O(d)$ multi-controlled rotation gates. You pay the exponential back in depth.
- **Normalization loss:** you must normalize $x$ first, so magnitude info is lost unless you add an extra qubit to encode the norm separately.
- **Gradient flow:** the input features $x_i$ enter as amplitudes, so $\partial f / \partial x_i$ has a different structure than for angle encoding — not always well-behaved.

PennyLane: `qml.AmplitudeEmbedding(x, wires=..., normalize=True)`.

### Rotations: gates → qubits → layers → circuits

This is the part people skip because the linear algebra feels heavy. Worth slowing down for.

**Single-qubit rotation gates.** Define the Pauli matrices $X = \begin{pmatrix}0&1\\1&0\end{pmatrix}$, $Y = \begin{pmatrix}0&-i\\i&0\end{pmatrix}$, $Z = \begin{pmatrix}1&0\\0&-1\end{pmatrix}$. The Pauli rotations are:
$$R_X(\theta) = e^{-i\theta X/2} = \cos(\theta/2) I - i\sin(\theta/2) X = \begin{pmatrix} \cos(\theta/2) & -i\sin(\theta/2) \\ -i\sin(\theta/2) & \cos(\theta/2)\end{pmatrix}.$$
Same shape for $R_Y$ and $R_Z$ with the corresponding Pauli. **Each is parametrized by exactly one angle** — this is the knob the optimizer turns.

Geometric picture: the qubit state on the *Bloch sphere* (a 2-sphere where $|0\rangle$ is north pole, $|1\rangle$ is south pole) gets rotated by angle $\theta$ around the corresponding Pauli axis. Three rotations are enough to reach any pure state.

**The general single-qubit rotation** `qml.Rot(φ, θ, ω)`:
$$\text{Rot}(\phi, \theta, \omega) = R_Z(\omega)\, R_Y(\theta)\, R_Z(\phi).$$
Three angles, covers all of $\text{SU}(2)$. This is the "fully expressive single-qubit unitary," and it's what our parity script uses.

**Rotations on $n$ qubits.** A single-qubit rotation on qubit $i$ is really $I \otimes \dots \otimes R \otimes \dots \otimes I$ — an $n$-fold tensor product where $R$ sits in the $i$-th slot. From the optimizer's point of view, each `qml.Rot(...)` on each qubit is **independent and contributes its own parameters**. Apply rotations to every qubit in parallel and you have $3n$ trainable angles in a single sweep, and no qubits are coupled yet.

**Entangling gates.** Without something to couple qubits, you have $n$ independent qubits with independent rotations and the model can only express product functions $f(x) = \prod_i f_i(x_i)$. You need entangling gates to make the joint state representation richer.

The standard choice: CNOT (controlled-NOT). Acts on two qubits — if the control is $|1\rangle$, flip the target; otherwise do nothing. After a CNOT, the two qubits are correlated: a measurement on one tells you something about the other.

**A "layer" = rotations + entangling pattern.** A typical strongly entangling layer (the term used by the QuKAN paper and our script):

```
for each qubit i:                  # 3n parameters per layer
    qml.Rot(θ[i,0], θ[i,1], θ[i,2], wires=i)
for each qubit i:                  # 0 parameters — fixed entangling pattern
    qml.CNOT(wires=[i, (i+1) % n])
```

**A "circuit" = encode + $L$ layers + measure.** Stack $L$ such layers and you have a circuit with $3nL$ trainable parameters. As $L$ grows the circuit can represent more complex functions — but you start running into barren plateaus and circuit-depth limits.

**The trainable parameters' role.** Every angle inside a `Rot` is just a NumPy scalar. The PennyLane autodiff (or the parameter-shift rule on hardware) computes $\partial L / \partial \theta$ for each one, and your optimizer (`NesterovMomentumOptimizer`, `AdamOptimizer`, ...) updates them like classical SGD. **From the optimizer's point of view this is identical to training a small neural net.** The only thing that's quantum is what $f_\theta(x)$ *means* — it's the expectation value coming out of a circuit instead of a NumPy matmul.

---

## Concept 3 — Training with vs. without bias

### The classical setup, transferred over
In logistic regression:
$$\hat y = \sigma(w \cdot x + b),$$
where $b$ is a scalar that shifts the decision boundary. Without it, the boundary $w \cdot x = 0$ is forced to pass through the origin in feature space — a real constraint when classes aren't symmetric about zero.

In the variational classifier:
$$f_\theta(x) = \langle Z_0 \rangle_{\theta, x} \in [-1, +1], \quad \hat y = \text{sign}\bigl(f_\theta(x) + b\bigr).$$
Same role: $b$ shifts the boundary from $\langle Z_0\rangle = 0$ to $\langle Z_0\rangle = -b$.

### Why and when bias matters
- **Class imbalance.** If 80% of your training labels are $+1$, the model's natural output should bias toward $+1$. Without a bias term, the quantum parameters have to absorb that shift by themselves — which wastes capacity that could have learned actual structure.
- **Output offset of the circuit at $\theta=0$.** With small random initialization, $\langle Z_0\rangle$ at $\theta \approx 0$ is *not* zero in general — it depends on the encoding $S(x)$. A bias lets you snap the model output to the right mean immediately, accelerating early training.
- **Cost:** bias is a single classical scalar parameter. No extra qubits, no extra gates, no extra circuit depth. **It is essentially free.**

### When you can get away without bias
- The labels are exactly balanced and centred at zero (e.g., $\pm 1$ parity labels).
- The encoding and ansatz happen to produce zero-mean outputs at initialization (e.g., basis encoding with random angles).
- You're doing a controlled experiment specifically about quantum expressivity.

For the parity task in `02_variational_classifier.py`, bias is *included* but the optimizer barely uses it: parity is perfectly balanced. You can verify this by checking `bias.item()` after training — it'll be tiny. Try setting `bias_init = np.array(0.0, requires_grad=False)` and re-running; on parity the result is essentially identical.

For moons or Iris (the QuKAN paper's benchmarks), bias matters more — those datasets aren't symmetric about zero in the encoded feature space.

### How the QuKAN paper handles "bias"
The QuKAN paper builds bias-like structure into the architecture itself, not as a separate scalar. Their KAN residual function is
$$\phi(x) = w_b \cdot \text{SiLU}(x) + w_s \sum_i \tilde c_i B_i(x),$$
where the $w_b\,\text{SiLU}(x)$ term acts as the model's "default activation" — a baseline the quantum spline part learns *corrections* to. In their ablation, when they remove the trainable quantum part entirely, the residual function reduces to "a scalable SiLU function and a bias term" — and the model still learns crude patterns. This is essentially saying: the bias-like baseline matters even without the quantum body.

### Practical recommendation
**Always include the bias term unless you're studying its absence in a controlled way.** One extra scalar is negligible. The case where you'd intentionally remove bias is in an ablation study like the one in the QuKAN paper: "how much of model performance comes from the quantum part, versus from the trivially-classical bias/activation?"

---

## Where to go next

1. Drop the arxiv PDF into your QML folder and ping me — I'll read it and add the summary here.
2. Try the experiments at the bottom of `02_variational_classifier.py`, especially #5 (swap to `AngleEmbedding`). That's the cleanest way to *feel* the encoding-choice difference rather than just read about it.
3. The QuKAN paper's GitHub repo is a candidate "well-engineered QML reference" for your project — if you want to study how a real QML model is implemented end-to-end. The paper cites the QKAN implementation at github.com/Mathewvanh/QKAN_Implementation.
