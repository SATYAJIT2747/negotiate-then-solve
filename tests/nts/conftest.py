import pytest

from nts.kernel.generator import generate_department
from nts.kernel.instance import policy_constraints
from nts.kernel.solver import TimetableSolver
from nts.state import ConstraintAdded, Store, TimetableSet


@pytest.fixture(scope="session")
def solved():
    """A small department and its semester timetable, solved once for the whole run."""
    inst = generate_department(seed=1, n_faculty=8, n_groups=3, n_lecture_rooms=4, n_labs=3)
    cons = policy_constraints(inst)
    result = TimetableSolver(inst, cons, time_limit=20).solve()
    assert result.ok
    return inst, cons, result.assignment


@pytest.fixture
def store(solved):
    """A fresh in-memory store at the solved timetable (no solve: tests may change it freely)."""
    inst, cons, assignment = solved
    return Store.create(inst, [*(ConstraintAdded(constraint=c) for c in cons), TimetableSet(assignment=assignment)])
