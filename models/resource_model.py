from dataclasses import dataclass
import math


@dataclass(frozen=True)
class User:
    name: str
    demand: float
    weight: float = 1.0
    disagreement_utility: float = 0.0

    def __post_init__(self):
        if not self.name.strip():
            raise ValueError("User name cannot be empty.")

        if not math.isfinite(self.demand) or self.demand < 0:
            raise ValueError("Demand must be finite and non-negative.")

        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("Weight must be finite and positive.")

        if (
            not math.isfinite(self.disagreement_utility)
            or self.disagreement_utility < 0
        ):
            raise ValueError("Disagreement utility must be non-negative.")

    def utility(self, allocation: float) -> float:
        if not math.isfinite(allocation) or allocation < -1e-9:
            raise ValueError("Allocation must be finite and non-negative.")

        allocation = max(0.0, allocation)

        # Weight now affects the curvature of utility.
        return math.log1p(self.weight * allocation)


def utility(user: User, allocation: float) -> float:
    return user.utility(allocation)


def validate_scenario(users, capacity):
    if not users:
        raise ValueError("At least one user is required.")

    if not math.isfinite(capacity) or capacity < 0:
        raise ValueError("Capacity must be finite and non-negative.")

    if len({user.name for user in users}) != len(users):
        raise ValueError("User names must be unique.")