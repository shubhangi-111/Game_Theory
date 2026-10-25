import math
from typing import Sequence

import numpy as np

from models.resource_model import (
    User,
    utility,
    validate_scenario,
)


def jain_fairness(values: Sequence[float]) -> float:
    """Jain's fairness index for non-negative values."""
    values = np.asarray(values, dtype=float)

    if values.size == 0:
        raise ValueError("At least one value is required.")

    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Values must be finite and non-negative.")

    denominator = len(values) * float(np.dot(values, values))

    if denominator == 0:
        return 1.0

    return float(values.sum() ** 2 / denominator)


def evaluate_allocation(
    users: Sequence[User],
    allocation: Sequence[float],
    capacity: float,
) -> dict:
    """Calculate utility, fairness and utilization metrics."""
    validate_scenario(users, capacity)

    allocation = np.asarray(allocation, dtype=float)

    if allocation.shape != (len(users),):
        raise ValueError("One allocation value is required per user.")

    if not np.all(np.isfinite(allocation)):
        raise ValueError("Allocations must be finite.")

    if np.any(allocation < -1e-7):
        raise ValueError("Negative allocations are not allowed.")

    if np.any(
        allocation > np.array([u.demand for u in users]) + 1e-6
    ):
        raise ValueError("An allocation exceeds a user's demand.")

    if allocation.sum() > capacity + 1e-6:
        raise ValueError("Total allocation exceeds available capacity.")

    allocation = np.maximum(allocation, 0.0)

    utilities = np.array([
        utility(user, x)
        for user, x in zip(users, allocation)
    ])

    gains = utilities - np.array([
        user.disagreement_utility for user in users
    ])

    # A positive Nash product requires strictly positive gains.
    if np.any(gains <= 0):
        log_nash_product = float("-inf")
        nash_product = 0.0
    else:
        log_nash_product = float(np.log(gains).sum())

        # Avoid overflow when reporting the product.
        nash_product = (
            math.exp(log_nash_product)
            if log_nash_product < 700
            else float("inf")
        )

    demands = np.array([u.demand for u in users])

    satisfaction = np.divide(
        allocation,
        demands,
        out=np.zeros_like(allocation),
        where=demands > 0,
    )

    eligible = demands > 0

    satisfaction_fairness = (
        jain_fairness(satisfaction[eligible])
        if np.any(eligible)
        else 1.0
    )

    return {
        "allocations": allocation.tolist(),
        "utilities": utilities.tolist(),
        "total_utility": float(utilities.sum()),
        "satisfaction_ratios": satisfaction.tolist(),
        "satisfaction_fairness": satisfaction_fairness,
        "allocation_fairness": jain_fairness(allocation),
        "resource_utilization": (
            float(allocation.sum() / capacity)
            if capacity > 0 else 0.0
        ),
        "allocated_capacity": float(allocation.sum()),
        "unused_capacity": float(capacity - allocation.sum()),
        "log_nash_product": log_nash_product,
        "nash_product": nash_product,
    }