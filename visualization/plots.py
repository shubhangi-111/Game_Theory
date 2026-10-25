from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def plot_allocations(
    users,
    allocations_by_strategy,
    output_dir="results",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    names = [user.name for user in users]
    x = np.arange(len(names))

    strategies = list(allocations_by_strategy.keys())
    width = 0.8 / len(strategies)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, strategy in enumerate(strategies):
        allocations = allocations_by_strategy[strategy]
        offset = (i - (len(strategies) - 1) / 2) * width

        ax.bar(
            x + offset,
            allocations,
            width,
            label=strategy,
        )

    ax.set_title("Resource Allocation by Strategy")
    ax.set_xlabel("Users")
    ax.set_ylabel("Allocated Resource Units")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        output_dir / "allocation_comparison.png",
        dpi=180,
    )
    plt.close(fig)


def plot_metrics(
    metrics_by_strategy,
    output_dir="results",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    strategies = list(metrics_by_strategy.keys())

    metric_specs = [
        ("total_utility", "Total Utility"),
        ("satisfaction_fairness", "Demand Satisfaction Fairness"),
        ("resource_utilization", "Resource Utilization"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (key, title) in zip(axes, metric_specs):
        values = [
            metrics_by_strategy[name][key]
            for name in strategies
        ]

        ax.bar(strategies, values)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        output_dir / "metrics_comparison.png",
        dpi=180,
    )
    plt.close(fig)