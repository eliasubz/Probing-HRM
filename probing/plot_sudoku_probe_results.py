import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt


def read_probe_csv(path):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "cycle": int(row["cycle"]),
                    "inner_call": int(row["inner_call"]),
                    "train_examples": int(row["train_examples"]),
                    "test_examples": int(row["test_examples"]),
                    "accuracy": float(row["accuracy"]),
                    "given_accuracy": float(row["given_accuracy"]),
                    "blank_accuracy": float(row["blank_accuracy"]),
                }
            )
    return rows


def group_by_inner_call(rows, metric):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["inner_call"]].append((row["cycle"], row[metric]))

    for inner_call in grouped:
        grouped[inner_call] = sorted(grouped[inner_call], key=lambda x: x[0])

    return grouped


def plot_metric(metric, title, out_path, linear_rows, mlp_rows):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, rows, label in [
        (axes[0], linear_rows, "Linear"),
        (axes[1], mlp_rows, "MLP"),
    ]:
        grouped = group_by_inner_call(rows, metric)
        for inner_call, values in sorted(grouped.items()):
            cycles = [v[0] for v in values]
            scores = [v[1] for v in values]
            ax.plot(cycles, scores, marker="o", label=f"inner_call={inner_call}")

        ax.set_title(f"{label} probe")
        ax.set_xlabel("cycle")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)

    axes[0].set_ylabel(metric)
    axes[1].legend(loc="lower right", fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "sudoku_probe_results")
    linear_path = os.path.join(base_dir, "h_layer3_linear_digit_probe.csv")
    mlp_path = os.path.join(base_dir, "h_layer3_mlp_digit_probe.csv")
    out_dir = os.path.join(base_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    linear_rows = read_probe_csv(linear_path)
    mlp_rows = read_probe_csv(mlp_path)

    plot_metric(
        "accuracy",
        "Digit probe accuracy by cycle",
        os.path.join(out_dir, "accuracy_by_cycle.png"),
        linear_rows,
        mlp_rows,
    )

    plot_metric(
        "blank_accuracy",
        "Blank-cell accuracy by cycle",
        os.path.join(out_dir, "blank_accuracy_by_cycle.png"),
        linear_rows,
        mlp_rows,
    )

    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()