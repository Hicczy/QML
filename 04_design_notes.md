# Comparison study: design notes

What this experiment compares: **two quantum encodings × three classifier families on grayscale image classification**, with accuracy and macro-F1 as metrics.

## The math your instructor flagged

Working through the formula step by step, since the whole design hinges on it.

Start with one qubit in the ground state:
$$|0\rangle = \begin{pmatrix} 1 \\ 0 \end{pmatrix}.$$

Apply the rotation $R_Y(\theta)$. PennyLane's convention is
$$R_Y(\theta) = \begin{pmatrix} \cos(\theta/2) & -\sin(\theta/2) \\ \sin(\theta/2) & \cos(\theta/2) \end{pmatrix}.$$

Multiplying it onto $|0\rangle$:
$$R_Y(\theta)|0\rangle = \begin{pmatrix} \cos(\theta/2) \\ \sin(\theta/2) \end{pmatrix} = \cos(\theta/2)\,|0\rangle + \sin(\theta/2)\,|1\rangle.$$

The Born rule says the probability of measuring the outcome $|1\rangle$ is the squared modulus of its amplitude:
$$P(|1\rangle) = |\langle 1 | R_Y(\theta) | 0\rangle|^2 = \sin^2(\theta/2).$$

Invert this to get the rotation that *produces* a desired probability:
$$\boxed{\theta = 2 \arcsin(\sqrt{P}).}$$

**Why this matters for "qubit as pixel" encoding.** If your normalized pixel value $p \in [0,1]$ should *be* the probability of measuring $|1\rangle$ on that qubit, then the rotation angle to encode it is $\theta = 2\arcsin(\sqrt{p})$. Bright pixel ($p \approx 1$) → $\theta \approx \pi$ → qubit ends up close to $|1\rangle$. Dark pixel ($p \approx 0$) → $\theta \approx 0$ → qubit stays in $|0\rangle$. Mid-gray → equal superposition.

This is the **principled angle encoding** we use in the experiment. A common shortcut is the linear map $\theta = \pi p$, which is approximately the same near $p = 0$ and $p = 1$ but differs in the middle. The arcsin version is the one that makes "the pixel value *is* the probability" literally true.

## Pipeline shape

For each of the 2 encodings × 3 models = 6 experiments, the pipeline is the same shape:

```
8x8 grayscale image
        │  (avg-pool 2x2)
        ▼
4x4 grayscale image (16 pixels)
        │  (encode + entangle + measure on PennyLane simulator)
        ▼
quantum feature vector   ← angle: 16 features (16 qubits)
                         ← amplitude: 4 features (4 qubits)
        │  (classical training)
        ▼
classifier prediction   ← MLP / SVM / KAN
```

The **quantum part is fixed** (no trainable circuit parameters). It's a deterministic feature extractor. Only the classical model is trained. This isolates "what does the encoding give us?" cleanly — any accuracy difference between encodings, holding the classifier fixed, comes purely from the encoding choice.

### The quantum feature extractor in detail

1. **Encode** the image into a quantum state $|\psi(x)\rangle$:
   - *Angle encoding*: for each pixel $p_i$, apply $R_Y(2\arcsin(\sqrt{p_i}))$ to qubit $i$. Each qubit's $\langle Z \rangle$ before entanglement is exactly $1 - 2p_i$ — a 1-1 linear map of the pixel.
   - *Amplitude encoding*: L2-normalize the flattened 16-pixel vector and load it into the amplitudes of a 4-qubit state. Uses `qml.AmplitudeEmbedding`.

2. **Entangle** with a single ring of CNOTs (no parameters). Without this, angle encoding's $\langle Z_i\rangle$ on each qubit would just be a deterministic function of pixel $i$ alone — no inter-pixel structure. With it, $\langle Z_i\rangle$ depends on multiple pixels through quantum correlations.

3. **Measure** $\langle Z_i \rangle$ on each qubit. This gives:
   - Angle: 16 real numbers in $[-1, 1]$
   - Amplitude: 4 real numbers in $[-1, 1]$

That vector is the input to the classical model.

### Dimensionality asymmetry — is it fair?

The two encodings naturally produce different feature dimensions (16 vs. 4). This *is* a real property of the encodings:
- Angle encoding uses one qubit per feature: $d$ features → $d$ qubits → $d$ measurements.
- Amplitude encoding packs $d$ features into $\log_2 d$ qubits, so only $\log_2 d$ single-qubit measurements are available.

I keep this asymmetry rather than artificially padding, because it's informative: it shows that amplitude encoding is *quantum-compact* in qubit count but *measurement-poor* (you've packed lots of features into very few qubits, so single-qubit observables can only tell you so much). The bigger feature set for angle encoding doesn't automatically mean better — it depends on whether those features carry distinct information.

### Why a fixed entangler, not a trainable one?

A trainable variational circuit on top of the encoding would be more powerful but would conflate two effects: encoding quality and ansatz expressivity. For a *comparison* study, we want one knob at a time. Trainable circuits are a natural next experiment.

## The three classical heads

- **MLP** — `sklearn.neural_network.MLPClassifier` with one hidden layer of 32 units, ReLU, Adam. The familiar feedforward baseline.
- **SVM** — `sklearn.svm.SVC` with RBF kernel, $C=1$. Strong non-linear classical baseline; works well in small-feature regimes.
- **KAN** — implemented in pure NumPy: each input feature $x_i$ is expanded into $K=8$ Gaussian radial basis functions, then a multinomial logistic regression sits on top. This is mathematically a **single-layer KAN with RBF basis functions on edges**. Each "edge" (input-feature × output-class pair) carries a learnable univariate function (the linear combination of RBFs), exactly the KAN architecture from Liu et al. 2024 (and the basis-function-on-edges structure from the QuKAN paper). I use RBFs instead of B-splines because they're trivial in NumPy; the *structural* property KAN cares about — learnable activations on edges — is preserved.

A real KAN can be multi-layer (alternating sum-then-univariate-nonlinearity), and the QuKAN paper itself uses 2 layers. Single-layer is the right place to start for a comparison — it isolates the "basis-functions-on-edges" idea cleanly without the additional confound of depth.

## Dataset

`sklearn.datasets.load_digits()`: 1797 samples, 8×8 grayscale, 10 classes (digits 0–9). I downsample to 4×4 via 2×2 average pooling.

Why 4×4: angle encoding uses one qubit per pixel. 4×4 = 16 qubits is comfortably simulable on a state-vector simulator. 8×8 = 64 qubits is well beyond simulator reach (would need a real quantum computer). Downsampling loses signal but keeps the task meaningful — accuracy on 4×4 should still be well above chance (10%).

## Evaluation

5-fold cross-validation. Report mean accuracy and mean macro-F1 across folds. Single-seed (seed=42) for reproducibility. Print a 6-row table at the end.

## What this study can and can't tell us

It **can** answer: under fixed simple-quantum-feature-extractor conditions, do angle and amplitude encoding produce features that classical models can usefully learn from, and which classical head pairs best with each?

It **can't** answer: would a trainable quantum classifier beat a classical model on this task? (Almost certainly not on digits — but that's not the question. The right question is methodological: do you understand each piece.)

Files:
- `quantum_encoding.py` — encoding + measurement, the only "quantum" code
- `simple_kan.py` — RBF-KAN classifier (sklearn-compatible)
- `05_compare_models.py` — runs the experiment, prints the comparison table, saves `results.csv`
