import pytest

from models.resource_model import User
from algorithms.baseline import equal_share, demand_proportional
from algorithms.nash_bargaining import solve_nash_bargaining
from evaluation.metrics import evaluate_allocation, jain_fairness


@pytest.fixture
def users():
    return [
        User("A", demand=10.0, weight=1.0),
        User("B", demand=10.0, weight=1.0),
        User("C", demand=10.0, weight=1.0),
    ]


@pytest.mark.parametrize(
    "algorithm",
    [equal_share, demand_proportional],
)
def test_baseline_capacity(algorithm, users):
    allocation = algorithm(users, 12.0)

    assert sum(allocation) <= 12.0 + 1e-6
    assert all(x >= 0 for x in allocation)

    for user, x in zip(users, allocation):
        assert x <= user.demand + 1e-6


def test_equal_sharing_is_equal_when_demands_match(users):
    allocation = equal_share(users, 12.0)

    assert allocation == pytest.approx([4.0, 4.0, 4.0])


def test_nash_symmetric_allocation(users):
    result = solve_nash_bargaining(users, 12.0)

    assert result["success"]
    assert result["allocation"] == pytest.approx(
        [4.0, 4.0, 4.0],
        abs=1e-3,
    )


def test_abundant_capacity_satisfies_all_demands(users):
    result = solve_nash_bargaining(users, 40.0)

    assert result["allocation"] == pytest.approx(
        [10.0, 10.0, 10.0],
        abs=1e-3,
    )


def test_zero_capacity():
    users = [User("A", demand=5.0), User("B", demand=5.0)]

    result = solve_nash_bargaining(users, 0.0)

    assert result["allocation"] == [0.0, 0.0]


def test_zero_demand_user():
    users = [
        User("A", demand=0.0),
        User("B", demand=10.0),
    ]

    result = solve_nash_bargaining(users, 5.0)

    assert result["allocation"][0] == 0.0
    assert result["allocation"][1] == pytest.approx(5.0, abs=1e-3)


def test_capacity_exceeded_is_rejected():
    users = [User("A", 5.0), User("B", 5.0)]

    with pytest.raises(ValueError):
        evaluate_allocation(users, [4.0, 4.0], 7.0)


def test_invalid_disagreement_utility_is_rejected():
    users = [
        User("A", demand=5.0, weight=1.0, disagreement_utility=2.0)
    ]

    with pytest.raises(ValueError):
        solve_nash_bargaining(users, 5.0)


def test_jain_fairness():
    assert jain_fairness([4.0, 4.0, 4.0]) == pytest.approx(1.0)

    assert jain_fairness([1.0, 0.0]) == pytest.approx(0.5)