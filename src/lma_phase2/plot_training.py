"""Create repository-ready, fully labelled Phase 2 loss and perplexity curves."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_log(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected LABEL=/path/to/training_log.csv")
    label, path = value.split("=", 1)
    return label, Path(path)


def read_log(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def plot_training_curves(logs: list[tuple[str, Path]], output: Path) -> None:
    """Plot train/validation CE and PPL for one or both independent models."""

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for label, path in logs:
        rows = read_log(path)
        for split, linestyle in (("train", "-"), ("validation", "--")):
            selected = [row for row in rows if row["split"] == split]
            if not selected:
                continue
            steps = [int(row["step"]) for row in selected]
            axes[0].plot(
                steps,
                [float(row["loss"]) for row in selected],
                linestyle,
                label=f"{label} {split}",
            )
            axes[1].plot(
                steps,
                [float(row["perplexity"]) for row in selected],
                linestyle,
                label=f"{label} {split}",
            )
    axes[0].set_title("Language-model cross-entropy during pretraining")
    axes[0].set_xlabel("Optimizer step")
    axes[0].set_ylabel("Cross-entropy loss (nats/token)")
    axes[1].set_title("Language-model perplexity during pretraining")
    axes[1].set_xlabel("Optimizer step")
    axes[1].set_ylabel("Perplexity")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", action="append", required=True, type=parse_log)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    plot_training_curves(args.log, args.output)


if __name__ == "__main__":
    main()
