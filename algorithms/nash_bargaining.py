import math
import numpy as np
from scipy.optimize import minimize

from models.resource_model import User, utility, validate_scenario


def solve_nash_bargaining(users: list[User], capacity: float) -> dict:
    validate_scenario(users, capacity)

    n = len(users)
    allocation = np.zeros(n)

    active_indices = [
        i for i, user in enumerate(users) if user.demand > 0
    ]

    if capacity == 0 or not active_indices:
        return {
            "allocation": allocation.tolist(),
            "success": True,
            "message": "No usable capacity or demand.",
            "log_nash_product": float("-inf"),
        }

    active = [users[i] for i in active_indices]
    demands = np.array([u.demand for u in active])

    # Minimum allocation needed to exceed disagreement utility.
    lower = np.array([
        np.expm1(u.disagreement_utility) / u.weight
        for u in active
    ])

    lower += 1e-8 * np.maximum(1.0, demands)

    if np.any(lower > demands) or lower.sum() > capacity + 1e-10:
        raise ValueError(
            "Capacity cannot satisfy every active user's "
            "disagreement utility."
        )

    x0 = lower.copy()
    remaining = min(capacity - x0.sum(), (demands - x0).sum())
    room = demands - x0

    if remaining > 0 and room.sum() > 0:
        x0 += remaining * room / room.sum()

    def objective(x):
        gains = np.array([
            utility(user, value) - user.disagreement_utility
            for user, value in zip(active, x)
        ])

        if np.any(gains <= 0) or not np.all(np.isfinite(gains)):
            return 1e100

        return -float(np.log(gains).sum())

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=list(zip(lower, demands)),
        constraints=[{
            "type": "ineq",
            "fun": lambda x: capacity - x.sum(),
        }],
        options={"maxiter": 2000, "ftol": 1e-10},
    )

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    x = result.x

    if (
        not np.all(np.isfinite(x))
        or np.any(x < -1e-7)
        or np.any(x > demands + 1e-6)
        or x.sum() > capacity + 1e-6
    ):
        raise RuntimeError("Optimizer returned an infeasible allocation.")

    for i, value in zip(active_indices, x):
        allocation[i] = max(0.0, float(value))

    return {
        "allocation": allocation.tolist(),
        "success": True,
        "message": str(result.message),
        "log_nash_product": -float(result.fun),
    }