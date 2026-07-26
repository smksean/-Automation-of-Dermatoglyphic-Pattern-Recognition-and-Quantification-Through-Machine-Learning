"""Generate a publication-ready comparison of aggregate experiment metrics."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main() -> None:
    summary = pd.read_csv(RESULTS / "experiment_summary.csv")
    labels = summary["experiment"].tolist()
    accuracy = summary["accuracy"].mul(100)
    macro_f1 = summary["macro_f1"].mul(100)

    figure, axis = plt.subplots(figsize=(10, 5.8))
    positions = range(len(summary))
    width = 0.36

    left = [position - width / 2 for position in positions]
    right = [position + width / 2 for position in positions]
    accuracy_bars = axis.bar(left, accuracy, width, label="Accuracy", color="#24557a")
    f1_bars = axis.bar(right, macro_f1, width, label="Macro F1", color="#d28c28")

    axis.set_title("Aggregate Broad-Pattern Classification Performance")
    axis.set_ylabel("Score (%)")
    axis.set_xticks(list(positions), labels, rotation=15, ha="right")
    axis.set_ylim(0, 100)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, loc="upper left")
    axis.bar_label(accuracy_bars, fmt="%.1f", padding=3, fontsize=9)
    axis.bar_label(f1_bars, fmt="%.1f", padding=3, fontsize=9)
    figure.text(
        0.5,
        0.01,
        "Evaluation cohorts differ; values document experimental progression rather than a paired comparison.",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    figure.savefig(
        RESULTS / "figures" / "experiment_performance_comparison.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
