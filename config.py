from models.resource_model import User


CAPACITY = 10.0

USERS = [
    User(
        name="AI Research",
        demand=8.0,
        weight=1.5,
        disagreement_utility=0.0,
    ),
    User(
        name="Data Analytics",
        demand=6.0,
        weight=1.0,
        disagreement_utility=0.0,
    ),
    User(
        name="Student Project",
        demand=5.0,
        weight=1.2,
        disagreement_utility=0.0,
    ),
]