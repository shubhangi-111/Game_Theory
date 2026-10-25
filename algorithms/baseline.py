from typing import Sequence

from models.resource_model import User, validate_scenario


def equal_share(
    users: Sequence[User],
    capacity: float
) -> list[float]:
    """Allocate capacity equally, respecting individual demands."""
    validate_scenario(users, capacity)

    allocations = [0.0] * len(users)
    remaining = capacity
    active = [
        i for i, user in enumerate(users)
        if user.demand > 0
    ]

    tolerance = 1e-10

    while active and remaining > tolerance:
        share = remaining / len(active)
        saturated = []

        for i in active:
            room = users[i].demand - allocations[i]

            if room <= share + tolerance:
                allocations[i] += max(0.0, room)
                remaining -= max(0.0, room)
                saturated.append(i)

        if not saturated:
            for i in active:
                allocations[i] += share

            remaining = 0.0
            break

        active = [i for i in active if i not in saturated]

    return allocations


def demand_proportional(
    users: Sequence[User],
    capacity: float
) -> list[float]:
    """Allocate capacity in proportion to requested demand."""
    validate_scenario(users, capacity)

    total_demand = sum(user.demand for user in users)

    if total_demand == 0 or capacity == 0:
        return [0.0] * len(users)

    scale = min(1.0, capacity / total_demand)

    return [user.demand * scale for user in users]