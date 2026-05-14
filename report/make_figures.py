"""
Generate every figure used in the report PDF.

Outputs (all into ./figs/):
    circuit_angle.pdf       — angle encoding circuit (PennyLane render)
    circuit_amplitude.pdf   — amplitude encoding circuit
    pipeline.pdf            — high-level data → encoding → model schematic
    main_results.pdf        — clean vs noisy hold-out accuracy bars
    robustness.pdf          — accuracy drop from noise, per encoding
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrow
import pennylane as qml

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)
# Results files live one directory up
ROOT = os.path.dirname(HERE)


# ----------------------------------------------------------------------
# Circuit diagrams via PennyLane's matplotlib drawer
# ----------------------------------------------------------------------
def draw_angle_circuit():
    dev = qml.device("default.qubit", wires=4)   # 4 qubits for clarity

    @qml.qnode(dev)
    def circ(angles):
        for i in range(4):
            qml.RY(angles[i], wires=i)
        for i in range(4):
            qml.CNOT(wires=[i, (i + 1) % 4])
        return [qml.expval(qml.PauliZ(i)) for i in range(4)]

    angles = np.array([0.4, 1.2, 2.1, 0.7])
    fig, _ = qml.draw_mpl(circ, decimals=2, style="pennylane")(angles)
    fig.suptitle("Angle encoding (qubit-per-pixel; 4-qubit illustration)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "circuit_angle.pdf"), bbox_inches="tight")
    plt.close(fig)


def draw_amplitude_circuit():
    dev = qml.device("default.qubit", wires=4)   # 4 qubits = 16 amplitudes

    @qml.qnode(dev)
    def circ(image_vec):
        qml.AmplitudeEmbedding(image_vec, wires=range(4), normalize=True)
        for i in range(4):
            qml.CNOT(wires=[i, (i + 1) % 4])
        return [qml.expval(qml.PauliZ(i)) for i in range(4)]

    vec = np.arange(1, 17, dtype=float)
    vec /= np.linalg.norm(vec)
    fig, _ = qml.draw_mpl(circ, decimals=2, style="pennylane",
                          level="device")(vec)
    fig.suptitle("Amplitude encoding (16 pixels → 4 qubits)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "circuit_amplitude.pdf"),
                bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Pipeline schematic
# ----------------------------------------------------------------------
def draw_pipeline():
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.set_xlim(0, 11); ax.set_ylim(0, 3.5); ax.axis("off")

    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.05",
            facecolor=color, edgecolor="#222", linewidth=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=10)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.4, color="#222"))

    # Stage 1: raw image
    box(0.0, 1.0, 1.8, 1.2, "MNIST\n28×28 → 4×4\n(avg-pool)", "#FFE9B5")
    arrow(1.85, 1.6, 2.4, 1.6)

    # Stage 2: encoding (two branches)
    box(2.5, 2.05, 2.6, 1.0,
        "Angle encoding\n16 qubits, $R_Y(2\\arcsin\\sqrt{p_i})$",
        "#BFE3FF")
    box(2.5, 0.15, 2.6, 1.0,
        "Amplitude encoding\n4 qubits, $\\sum_i p_i\\,|i\\rangle$",
        "#BFE3FF")
    arrow(2.4, 1.6, 2.5, 2.55)
    arrow(2.4, 1.6, 2.5, 0.65)

    # Stage 3: CNOT ring + measure (shared)
    box(5.3, 1.0, 2.4, 1.2, "CNOT ring\n+ measure $\\langle Z_i\\rangle$",
        "#FFD0E0")
    arrow(5.15, 2.55, 5.3, 1.85)
    arrow(5.15, 0.65, 5.3, 1.35)

    # Stage 4: classical model
    box(7.9, 1.0, 2.6, 1.2, "Classical head\nLR / MLP / SVM / KAN",
        "#D6F2D0")
    arrow(7.75, 1.6, 7.9, 1.6)

    # Output
    box(10.55, 1.2, 0.4, 0.8, "$\\hat{y}$", "#FFFFFF")
    arrow(10.5, 1.6, 10.55, 1.6)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "pipeline.pdf"), bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Main results figure: clean vs noisy hold-out accuracy
# ----------------------------------------------------------------------
def draw_results():
    df = pd.read_csv(os.path.join(ROOT, "results_holdout.csv"))
    agg = df.groupby(["encoding", "model", "condition"], as_index=False).agg(
        accuracy=("accuracy", "mean"),
        f1=("f1_macro", "mean"),
        std=("accuracy", "std"),
    )

    enc_order   = ["raw", "angle", "amplitude"]
    model_order = ["LR", "MLP", "SVM", "KAN"]
    width = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, cond, title in [
        (axes[0], "clean",     "Clean hold-out test"),
        (axes[1], "noisy_0.2", "Noisy hold-out test ($\\sigma=0.2$)")
    ]:
        sub = agg[agg["condition"] == cond]
        pivot_acc = sub.pivot(index="encoding", columns="model", values="accuracy")
        pivot_std = sub.pivot(index="encoding", columns="model", values="std")
        pivot_acc = pivot_acc.loc[enc_order, model_order]
        pivot_std = pivot_std.loc[enc_order, model_order].fillna(0.0)
        x = np.arange(len(enc_order))
        for i, m in enumerate(model_order):
            ax.bar(x + (i - 1.5) * width, pivot_acc[m].values, width,
                   yerr=pivot_std[m].values, capsize=2.5,
                   label=m, edgecolor="#222", linewidth=0.4)
        ax.axhline(0.1, color="gray", linestyle="--", linewidth=0.9,
                   label="chance (10 cls)")
        ax.set_xticks(x); ax.set_xticklabels(enc_order)
        ax.set_ylim(0, 1.0); ax.set_title(title, fontsize=11)
        ax.set_xlabel("encoding")
    axes[0].set_ylabel("accuracy")
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "main_results.pdf"), bbox_inches="tight")
    plt.close(fig)


def draw_robustness():
    df = pd.read_csv(os.path.join(ROOT, "results_holdout.csv"))
    agg = df.groupby(["encoding", "model", "condition"], as_index=False).agg(
        accuracy=("accuracy", "mean"),
    )
    enc_order   = ["raw", "angle", "amplitude"]
    model_order = ["LR", "MLP", "SVM", "KAN"]
    # accuracy drop = clean - noisy
    clean = agg[agg["condition"] == "clean"].set_index(["encoding", "model"])["accuracy"]
    noisy = agg[agg["condition"] == "noisy_0.2"].set_index(["encoding", "model"])["accuracy"]
    drop = (clean - noisy).reset_index()
    pivot = drop.pivot(index="encoding", columns="model", values="accuracy")
    pivot = pivot.loc[enc_order, model_order]

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    x = np.arange(len(enc_order))
    width = 0.18
    for i, m in enumerate(model_order):
        ax.bar(x + (i - 1.5) * width, pivot[m].values, width,
               label=m, edgecolor="#222", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(enc_order)
    ax.set_ylabel("accuracy drop from $\\sigma=0.2$ noise")
    ax.set_xlabel("encoding")
    ax.set_title("Robustness: lower is better")
    ax.set_ylim(0, 0.75)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "robustness.pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    print("Drawing pipeline ..."); draw_pipeline()
    print("Drawing angle circuit ..."); draw_angle_circuit()
    print("Drawing amplitude circuit ..."); draw_amplitude_circuit()
    print("Drawing main results ..."); draw_results()
    print("Drawing robustness ..."); draw_robustness()
    print("Done. Figures in", FIGS)


if __name__ == "__main__":
    main()
