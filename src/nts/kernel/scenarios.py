"""Hand-built use-case scenarios (proposal Section 9) used as fixtures."""

from __future__ import annotations

from .instance import (
    Faculty,
    Group,
    Instance,
    Room,
    RoomType,
    Session,
    SessionKind,
    policy_constraints,
)
from .schemas import (
    Constraint,
    ConstraintType,
    Justification,
    RoomRequirement,
    Scope,
    Source,
    Tier,
    When,
)

AFTERNOON = [5, 6, 7]  # slots after the lunch slot (4)


def uc3_lab_contention() -> tuple[Instance, list[Constraint]]:
    """UC3: Dr. Khan (GPUs) and Dr. Das (routers) both need Lab 2 on Tuesday
    afternoon. Lab 2 is the only lab with both; Lab 4 has neither.

    Two 2-slot practicals cannot share three afternoon slots in one room, so
    the four Tier 4 constraints conflict. Expected MUS: all four; expected
    MCS: drop any one of them.
    """
    instance = Instance(
        name="uc3-lab-contention",
        rooms=[
            Room(id="R-101", name="Room 101", capacity=80, type=RoomType.LECTURE),
            Room(id="L-2", name="Lab 2", capacity=60, type=RoomType.LAB, equipment=["gpu", "routers"]),
            Room(id="L-4", name="Lab 4", capacity=60, type=RoomType.LAB),
        ],
        faculty=[
            Faculty(id="F-201", name="Dr. Khan", department="CSE"),
            Faculty(id="F-202", name="Dr. Das", department="CSE"),
        ],
        groups=[
            Group(id="G-A", name="CSE 3A", size=50),
            Group(id="G-B", name="CSE 3B", size=45),
        ],
        sessions=[
            Session(id="ML-P", course="ML", kind=SessionKind.PRACTICAL, faculty="F-201",
                    groups=["G-A"], duration=2, room_type=RoomType.LAB),
            Session(id="NET-P", course="NET", kind=SessionKind.PRACTICAL, faculty="F-202",
                    groups=["G-B"], duration=2, room_type=RoomType.LAB),
            Session(id="ML-L1", course="ML", kind=SessionKind.LECTURE, faculty="F-201", groups=["G-A"]),
            Session(id="NET-L1", course="NET", kind=SessionKind.LECTURE, faculty="F-202", groups=["G-B"]),
        ],
    )

    def lab_constraints(cid: str, owner: str, session: str, equipment: str, request: str) -> list[Constraint]:
        return [
            Constraint(
                id=f"{cid}-ROOM", type=ConstraintType.REQUIRE_ROOM, hard=True, tier=Tier.OPERATIONAL,
                owner=owner, scope=Scope(session=session), room=RoomRequirement(equipment=[equipment]),
                justification=Justification.STATED, source=Source(request=request),
            ),
            Constraint(
                id=f"{cid}-TIME", type=ConstraintType.PREFER, hard=True, tier=Tier.OPERATIONAL,
                owner=owner, scope=Scope(session=session), when=When(days=["Tue"], slots=AFTERNOON),
                justification=Justification.STATED, source=Source(request=request),
            ),
        ]

    constraints = (
        policy_constraints(instance)
        + lab_constraints("C-KHAN", "F-201", "ML-P", "gpu", "R-3001")
        + lab_constraints("C-DAS", "F-202", "NET-P", "routers", "R-3002")
    )
    return instance, constraints
