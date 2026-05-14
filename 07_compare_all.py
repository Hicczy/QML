"""
Three-way comparison summary:
   1. results.csv               — original sklearn-digits, single-seed,  no noise
   2. results_mnist_clean.csv   — MNIST, 3 seeds × 5-fold CV,            no noise
   3. results_mnist_noisy_0.2.csv — MNIST, 3 seeds × 5-fold CV, σ=0.2 noise

Prints accuracy/F1 tables side by side and saves a comparison plot.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def load_results():
    """Return three DataFrames in a consistent (encoding, model, accuracy_mean, f1_macro_mean) shape."""
    # 1. sklearn-digits (original)
    df1 = pd.read_csv(os.path.join(HERE, "results.csv"))[
        ["encoding", "model", "accuracy_mean", "f1_macro_mean"]
    ]
    df1["dataset"] = "sklearn-digits (clean)"

    # 2. MNIST clean — aggregate over seeds
    df2_raw = pd.read_csv(os.path.join(HERE, "results_mnist_clean.csv"))
    df2 = df2_raw.groupby(["encoding", "model"], as_index=False).agg(
        accuracy_mean=("accuracy_mean", "mean"),
        f1_macro_mean=("f1_macro_mean", "mean"),
    )
    df2["dataset"] = "MNIST (clean)"

    # 3. MNIST noisy σ=0.2
    df3_raw = pd.read_csv(os.path.join(HERE, "results_mnist_noisy_0.2.csv"))
    df3 = df3_raw.groupby(["encoding", "model"], as_index=False).agg(
        accuracy_mean=("accuracy_mean", "mean"),
        f1_macro_mean=("f1_macro_mean", "mean"),
    )
    df3["dataset"] = "MNIST (noisy σ=0.2)"

    return df1, df2, df3


def print_tables(df1, df2, df3):
    enc_order = ["raw", "angle", "amplitude"]
    model_order_full = ["LR", "KAN", "MLP", "SVM"]

    for name, df in [
        ("sklearn-digits (single-seed, no noise)", df1),
        ("MNIST clean (3 seeds, no noise)",        df2),
        ("MNIST noisy (3 seeds, σ=0.2)",           df3),
    ]:
        pivot = df.pivot(index="encoding", columns="model", values="accuracy_mean")
        models_present = [m for m in model_order_full if m in pivot.columns]
        encodings_present = [e for e in enc_order if e in pivot.index]
        print(f"\n--- {name} — accuracy ---")
        print(pivot.loc[encodings_present, models_present].round(3).to_string())


def plot_comparison(df2, df3, save_path):
    """Bar plot: MNIST clean vs MNIST noisy, grouped by (encoding, model)."""
    enc_order   = ["raw", "angle", "amplitude"]
    model_order = ["LR", "MLP", "SVM", "KAN"]

    # Build aligned arrays
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, df, title in [(axes[0], df2, "MNIST clean"),
                          (axes[1], df3, "MNIST noisy (σ=0.2)")]:
        pivot = df.pivot(index="encoding", columns="model", values="accuracy_mean")
        pivot = pivot.loc[enc_order, model_order]
        x = np.arange(len(enc_order))
        width = 0.18
        for i, model in enumerate(model_order):
            ax.bar(x + (i - 1.5) * width, pivot[model].values, width, label=model)
        ax.set_xticks(x); ax.set_xticklabels(enc_order)
        ax.set_ylabel("accuracy"); ax.set_title(title)
        ax.set_ylim(0, 1.0); ax.axhline(0.1, color="gray", linestyle="--",
                                         label="chance (10-class)")
        ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"\nSaved plot -> {save_path}")


def print_noise_robustness(df2, df3):
    """How much does each cell drop under noise?"""
    clean = df2.set_index(["encoding", "model"])["accuracy_mean"]
    noisy = df3.set_index(["encoding", "model"])["accuracy_mean"]
    delta = (noisy - clean).reset_index().rename(columns={"accuracy_mean": "delta"})
    pivot = delta.pivot(index="encoding", columns="model", values="delta")
    pivot = pivot.loc[["raw", "angle", "amplitude"], ["LR", "MLP", "SVM", "KAN"]]
    print("\n--- Accuracy drop from noise (noisy - clean) ---")
    print(pivot.round(3).to_string())
    print("\n(More negative = bigger drop = less robust)")


def main():
    df1, df2, df3 = load_results()
    print_tables(df1, df2, df3)
    print_noise_robustness(df2, df3)
    plot_comparison(df2, df3, os.path.join(HERE, "mnist_comparison.png"))


if __name__ == "__main__":
    main()
