import csv
from pathlib import Path

from config import CAPACITY, USERS

from algorithms.baseline import (
    equal_share,
    demand_proportional,
)

from algorithms.nash_bargaining import solve_nash_bargaining

from evaluation.metrics import evaluate_allocation

from visualization.plots import (
    plot_allocations,
    plot_metrics,
)


RESULTS_DIR = Path("results")


def save_csv(users, allocations_by_strategy, metrics_by_strategy):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Per-user allocation and utility results.
    with open(
        RESULTS_DIR / "user_allocations.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "User",
            "Strategy",
            "Demand",
            "Allocation",
            "Utility",
            "Demand_Satisfaction_Ratio",
        ])

        for strategy, allocations in allocations_by_strategy.items():
            metrics = metrics_by_strategy[strategy]

            for i, user in enumerate(USERS):
                writer.writerow([
                    user.name,
                    strategy,
                    user.demand,
                    round(allocations[i], 6),
                    round(metrics["utilities"][i], 6),
                    round(metrics["satisfaction_ratios"][i], 6),
                ])

    # Strategy-level comparison results.
    with open(
        RESULTS_DIR / "strategy_metrics.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "Strategy",
            "Total_Utility",
            "Satisfaction_Fairness",
            "Allocation_Fairness",
            "Resource_Utilization",
            "Allocated_Capacity",
            "Unused_Capacity",
            "Log_Nash_Product",
        ])

        for strategy, metrics in metrics_by_strategy.items():
            writer.writerow([
                strategy,
                metrics["total_utility"],
                metrics["satisfaction_fairness"],
                metrics["allocation_fairness"],
                metrics["resource_utilization"],
                metrics["allocated_capacity"],
                metrics["unused_capacity"],
                metrics["log_nash_product"],
            ])


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Run all three strategies on the same scenario.
    nash_result = solve_nash_bargaining(USERS, CAPACITY)

    allocations_by_strategy = {
        "Equal Sharing": equal_share(USERS, CAPACITY),
        "Demand Proportional": demand_proportional(
            USERS, CAPACITY
        ),
        "Nash Bargaining": nash_result["allocation"],
    }

    metrics_by_strategy = {
        strategy: evaluate_allocation(
            USERS, allocation, CAPACITY
        )
        for strategy, allocation in allocations_by_strategy.items()
    }

    print("\n" + "=" * 72)
    print("BARGAINING-BASED RESOURCE ALLOCATION SIMULATOR")
    print("=" * 72)

    print(f"Available capacity: {CAPACITY:.2f} units")
    print(f"Total requested demand: {sum(u.demand for u in USERS):.2f}")
    print(f"Nash solver: {nash_result['message']}")

    print("\nUSER ALLOCATIONS")
    print("-" * 72)
    print(
        f"{'User':<22}"
        f"{'Demand':>10}"
        f"{'Equal':>10}"
        f"{'Prop.':>10}"
        f"{'Nash':>10}"
    )

    for i, user in enumerate(USERS):
        print(
            f"{user.name:<22}"
            f"{user.demand:>10.2f}"
            f"{allocations_by_strategy['Equal Sharing'][i]:>10.2f}"
            f"{allocations_by_strategy['Demand Proportional'][i]:>10.2f}"
            f"{allocations_by_strategy['Nash Bargaining'][i]:>10.2f}"
        )

    print("\nSTRATEGY COMPARISON")
    print("-" * 90)
    print(
        f"{'Strategy':<23}"
        f"{'Utility':>12}"
        f"{'Sat. Fairness':>16}"
        f"{'Utilization':>15}"
    )

    for strategy, metrics in metrics_by_strategy.items():
        print(
            f"{strategy:<23}"
            f"{metrics['total_utility']:>12.4f}"
            f"{metrics['satisfaction_fairness']:>16.4f}"
            f"{metrics['resource_utilization']:>14.2%}"
        )

    print("\nINDIVIDUAL UTILITIES")
    for strategy, metrics in metrics_by_strategy.items():
        print(f"\n{strategy}:")

        for user, value in zip(USERS, metrics["utilities"]):
            print(f"  {user.name}: {value:.4f}")

    plot_allocations(
        USERS,
        allocations_by_strategy,
        RESULTS_DIR,
    )

    plot_metrics(
        metrics_by_strategy,
        RESULTS_DIR,
    )

    save_csv(
        USERS,
        allocations_by_strategy,
        metrics_by_strategy,
    )

    print("\nFiles generated in the results/ directory:")
    print("  - allocation_comparison.png")
    print("  - metrics_comparison.png")
    print("  - user_allocations.csv")
    print("  - strategy_metrics.csv")


if __name__ == "__main__":
    main()