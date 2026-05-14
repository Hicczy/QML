"""
Quantum encodings + measurement-based feature extraction.

Two encodings, both followed by a single CNOT ring (fixed, non-trainable
entangling layer) and then ⟨Z_i⟩ measured on every qubit.

   image (4x4=16 pixels in [0,1])
        │
        ▼  angle:        amplitude:
   16 qubits          4 qubits
   one R_Y(2*asin(√p)) AmplitudeEmbedding(image,
   per qubit                normalize=True)
        │
        ▼
   CNOT ring (i → i+1)
        │
        ▼
   measure ⟨Z_i⟩ for each qubit
        │
        ▼
   feature vector (16 for angle, 4 for amplitude)

Use:
    features_angle = encode_dataset(X, kind="angle")
    features_amp   = encode_dataset(X, kind="amplitude")
"""

from __future__ import annotations
import numpy as np
import pennylane as qml


# ----------------------------------------------------------------------
# Pixel → angle map, exactly the formula in 04_design_notes.md.
# pixel value p ∈ [0,1] is mapped to the rotation angle θ such that
# P(measure |1>) = sin²(θ/2) = p. So θ = 2·arcsin(√p).
# ----------------------------------------------------------------------
def pixel_to_angle(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 0.0, 1.0)
    return 2.0 * np.arcsin(np.sqrt(p))


# ----------------------------------------------------------------------
# Angle encoding circuit
# ----------------------------------------------------------------------
def _make_angle_circuit(n_qubits: int):
    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(angles):
        # 1. Encode: one R_Y per pixel.
        for i in range(n_qubits):
            qml.RY(angles[i], wires=i)
        # 2. Fixed entangler: ring of CNOTs.
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])
        # 3. Read out ⟨Z_i⟩ on every qubit.
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


# ----------------------------------------------------------------------
# Amplitude encoding circuit
# ----------------------------------------------------------------------
def _make_amplitude_circuit(n_qubits: int):
    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(image_vec):
        # 1. Encode: load normalized image into 2^n amplitudes.
        qml.AmplitudeEmbedding(image_vec, wires=range(n_qubits), normalize=True)
        # 2. Fixed entangler: ring of CNOTs.
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])
        # 3. Read out ⟨Z_i⟩.
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def encode_dataset(X: np.ndarray, kind: str, verbose: bool = True) -> np.ndarray:
    """
    Apply the chosen encoding to every row of X.

    Args:
        X: array (N, n_pixels) of values in [0, 1].
        kind: 'angle' or 'amplitude'.
        verbose: print a progress line every 250 samples.

    Returns:
        features: array (N, n_qubits) of expectation values in [-1, 1].
    """
    N, n_pixels = X.shape

    if kind == "angle":
        n_qubits = n_pixels
        circuit = _make_angle_circuit(n_qubits)
        inputs = pixel_to_angle(X)
    elif kind == "amplitude":
        n_qubits = int(np.ceil(np.log2(n_pixels)))
        # Pad image to length 2^n_qubits if needed.
        target_len = 2 ** n_qubits
        if n_pixels < target_len:
            pad = np.zeros((N, target_len - n_pixels))
            inputs = np.concatenate([X, pad], axis=1)
        else:
            inputs = X
        circuit = _make_amplitude_circuit(n_qubits)
    else:
        raise ValueError(f"unknown encoding: {kind!r}")

    features = np.zeros((N, n_qubits))
    for i in range(N):
        features[i] = np.asarray(circuit(inputs[i]))
        if verbose and (i + 1) % 250 == 0:
            print(f"  encoded {i + 1}/{N}")

    return features
