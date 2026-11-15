"""Orchestrator: the request lifecycle (proposal Figure 2) end to end.

Received -> Classified (System One routes: fast path to the compiler, or
System Two) -> Policy checked (RAG) -> Compiled + validated -> Solved
(CP-SAT repair) -> Negotiating (resolution ladder) -> Fairness audited ->
Awaiting approval -> Published + notified.

Safe exits: Refused (no authority), Denied (policy, with citation and
alternative), Clarification requested, Escalated, Answered, Forwarded.

Safety properties enforced here, not by any model:

* nothing is published except by ``approve`` from a listed approver;
* message text never reaches a state-changing call: models return data,
  and only this code decides what happens with it;
* replies and notices are built from constraints, never from private reasons;
* requests are compiled and solved one at a time (``_solving``): each builds on the change before it,
  and the solver reads every constraint in the store, so two requests solved together would each
  carry the other's change (two absences in one week, each cancelling both teachers' classes).
"""

from __future__ import annotations

import contextlib
import functools
import re
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime

from agents.explainer import _when, describe, placement_text, session_name
from agents.extra import RULE as EXTRA_RULE
from agents.extra import (
    TEACHERS,
    ExtraReader,
    ExtraReading,
    block_constraints,
    extra_session,
    free_placements,
    looks_like_extra,
    when_text,
)
from agents.ledger import ConcessionLedger
from agents.negotiation import EscalationBrief, Message, Negotiator, Offer, Outcome, Responder
from agents.policy import PolicyAgent, PolicyDecision, Verdict
from agents.swap import (
    RuleSwapReader,
    SwapPlan,
    SwapReading,
    check_swap,
    consent_message,
    looks_like_swap,
    swap_constraints,
)
from core.graph import add_change, affected_stakeholders, build_graph, moved_sessions
from core.instance import ExtraClass, Instance
from core.schemas import (
    DAYS,
    Constraint,
    ConstraintType,
    Placement,
    Request,
    RequestStatus,
    Role,
    Scope,
    Source,
    Tier,
    TimetableVersion,
    When,
)
from core.semantics import PLACEMENT_TYPES, room_compatible, room_satisfies, sessions_in_scope, violates
from core.solver import TimetableSolver
from core.validator import validate_constraint, verify_timetable
from language.shorthand import expand as expand_shorthand
from language.corpus import FULL_DAY, ExpectedAction, RequestType, hour
from language.intents import (
    AnyOf,
    Avoid,
    ClashReport,
    ClubRef,
    Compiled,
    IntentOutput,
    Limit,
    LoadLimit,
    Move,
    NeedsRoom,
    Only,
    Ranked,
    Relate,
    RoomClosed,
    Unavailable,
    Unresolved,
    classes,
    compile_intents,
    say,
    unaccounted,
)
from language.intents import Cancel as CancelIntent
from language.intents import ClubBooking as ClubIntent
from language.intents import ExtraClass as ExtraIntent
from language.intents import Fallback as FallbackIntent
from language.intents import Swap as SwapIntent
from language.parsing import STUDENTS_REPORT, ParseResult
from language.rule_parser import RuleParser, injected, request_part, surname
from pipeline.exams import exam_ruling

from .store import EXTRA_PREFIX, Store

EXAM_CELL = "exam_cell"  # escalated to the role, like "hod" and "dean": whoever holds it decides


S = RequestStatus
CHOICES = 3  # times offered to a sender whose request fits in more than one way
# intents that become constraints: one message's are solved together, as one change
CONSTRAINT_INTENTS = (Unavailable, Avoid, Only, Move, RoomClosed, NeedsRoom, LoadLimit, Limit, Relate)


class NotAuthorised(PermissionError):
    pass


@dataclass
class Case:
    request: Request
    route: str = ""
    parse: ParseResult | None = None
    policy: PolicyDecision | None = None
    constraints: list[Constraint] = field(default_factory=list)
    outcome: Outcome | None = None
    proposal: TimetableVersion | None = None
    fairness: dict = field(default_factory=dict)
    reply: str = ""
    notices: dict[str, str] = field(default_factory=dict)
    routing: dict = field(default_factory=dict)  # System One decision, when routing is on
    timings: dict[str, float] = field(default_factory=dict)  # seconds per stage
    superseded: list[str] = field(default_factory=list)  # earlier constraints this request replaced
    decision: dict | None = None  # how a person with the authority settled its escalation
    handled: dict | None = None  # the coordinator's answer to a message forwarded to them
    extra: dict | None = None  # an extra class: which class and week, and where it is held once found
    compiled: Compiled | None = None  # the resolved intents, when the parser reads intents
    form: IntentOutput | None = None  # the request as made with clicks (a form): read without a parser
    more: list[TimetableVersion] = field(default_factory=list)  # the proposals for its further weeks, in order
    read_as: Request | None = None  # the request as its readers see it: lines instructing the system left out
    parts: list[str] = field(default_factory=list)  # the requests a message of several parts was split into
    understood: list[str] = field(default_factory=list)  # each intent read from the message, in words
    unused: list[str] = field(default_factory=list)  # what the message says that the reading does not use

    def intents(self, kind: type) -> list:
        """The parsed intents of type ``kind`` (none for a parser without intents) that the request does now: not
        the members of a choice, which go with it, nor a fallback's second item, which waits on the first."""
        out = self.compiled.output if self.compiled is not None else None
        if out is None or out.action != "compile":
            return []
        held = {x for i in out.intents if isinstance(i, (AnyOf, Ranked)) for x in i.of}
        held |= {i.of[1] for i in out.intents if isinstance(i, FallbackIntent)}
        return [i for i in out.intents if isinstance(i, kind) and (getattr(i, "id", None) not in held)]

    @property
    def id(self) -> str:
        return self.request.id

    @property
    def status(self) -> RequestStatus:
        return self.request.status

    @property
    def forwarded_to(self) -> str | None:
        """Who a forwarded message went to: "coordinator", or "office" (outside the timetable)."""
        if self.status != RequestStatus.FORWARDED:
            return None
        return "office" if self.parse and self.parse.action == ExpectedAction.OUT_OF_SCOPE else "coordinator"


def closures(instance: Instance, constraints: Iterable[Constraint], week: int | None) -> tuple[set[str], list[str]]:
    """Classes a room closure leaves nowhere to go in ``week``, and the closed rooms. A lab in-charge
    reporting "Lab 3 is closed in week 9" states a fact, not a request anyone can turn down: when every
    room a class can use is closed for that whole week, the class is cancelled for the week (a make-up is
    owed) rather than left in a closed room or the closure escalated. The rooms a class can use are those
    its own equipment allows and, as well, any hard room requirement in force that week ("my practical
    needs the routers"): such a need is not something the office can grant away against a closure."""
    if week is None:
        return set(), []
    constraints = list(constraints)
    closed = {c.scope.room for c in constraints
              if c.hard and c.type == ConstraintType.UNAVAILABLE and c.scope.room and c.active_in(week)
              and (c.when.weeks is None or week in c.when.weeks) and not c.when.days and not c.when.slots}
    if not closed:
        return set(), []
    needs: dict[str, list] = {}  # session -> the hard room requirements on it that week
    for c in constraints:
        if (c.hard and c.type == ConstraintType.REQUIRE_ROOM and c.room is not None and c.active_in(week)
                and (c.when.weeks is None or week in c.when.weeks)):
            for s in sessions_in_scope(instance, c.scope):
                needs.setdefault(s.id, []).append(c.room)
    out = set()
    for s in instance.sessions:
        rooms = [r.id for r in instance.rooms if room_compatible(instance, s, r)
                 and all(room_satisfies(req, r) for req in needs.get(s.id, []))]
        if rooms and set(rooms) <= closed:
            out.add(s.id)
    return out, sorted(instance.room_by_id[r].name for r in closed)


# a teacher telling the system to change a named colleague's class: "move Dr. Iyer's ...", "cancel Dr. Rao's ..."
ORDERS_COLLEAGUE = re.compile(r"\b(?:move|moving|cancel|cancelling|shift|reschedule|postpone|drop|put|give)\b"
                              r"\s+(?:(?:the|please|kindly|also|just)\s+)*(?:dr|prof)\.?\s+([a-z]+)'s\b", re.IGNORECASE)


def _one_at_a_time(method):
    """Run ``method`` holding the orchestrator's ``_solving`` lock (see the module docstring)."""
    @functools.wraps(method)
    def run(self, *args, **kwargs):
        with self._solving:
            return method(self, *args, **kwargs)
    return run


# a message taking back an earlier request says so: a reading of "withdraw" for anything else is not acted on
SAYS_WITHDRAW = re.compile(
    r"\b(withdraw\w*|retract\w*|take (?:it|that|this|them) back|take back|never ?mind|nevermind|scrap|"
    r"forget (?:it|about|that|my|the)|ignore (?:my|that|the|this)\b|disregard|"
    r"(?:cancel|drop|delete|remove) (?:my|that|the|this) (?:\w+ )?(?:request|message)|"
    r"(?:no longer|don't|do not) need (?:it|that|this|the change)|not needed any ?more)",
    re.IGNORECASE)

class Orchestrator:
    def __init__(
        self,
        instance: Instance,
        store: Store,
        *,
        parser,
        negotiator: Negotiator,
        fast_parser=None,
        system_one=None,
        tau: float = 0.9,
        use_fast_path: Callable[[], bool] | None = None,
        policy_agent: PolicyAgent | None = None,
        form_policy=None,
        responders: Mapping[str, Responder] | None = None,
        approvers: Iterable[str] = ("C-TT",),
        semester: str = "current",
        swap_reader=None,
        offer_choices: Callable[[], bool] | None = None,
        current_week: Callable[[], int | None] | None = None,
        today: Callable[[], date] | None = None,
        club_desk: Callable[[ClubRef, Request], tuple[str, str]] | None = None,
    ) -> None:
        """``offer_choices``: whether, right now, a sender whose request fits in more than one way is
        asked which time they prefer (``_let_sender_choose``); never when omitted. ``current_week``: the
        teaching week now, so a parser reading intents can resolve "next week"; unknown when omitted.

        A parser with ``compile(request, timetable, current_week)`` reads intents (``language.intents``):
        swaps, extra classes, cancellations and withdrawals are routed by what it read. ``compile`` may
        return None (a stand-in without intents answers): the request is then parsed as before.

        ``use_fast_path``: whether, right now, a request System One is sure of may go to ``fast_parser``
        (always when omitted). The web app takes it only while the model is unreachable: the fine-tuned reader
        reads far better than the fast parser, which System One's confidence does not show.

        ``club_desk`` books a club activity from a message (``ClubRef``) and returns the reply and the outcome
        (booked, rejected, or needs_approval: the timetable office decides); without one, club bookings go to
        the coordinator.

        ``form_policy`` checks requests made with clicks (``Case.form``) in place of ``policy_agent``: their
        intents are exact, so the rule-based check (``RulePolicyAgent``) suffices and no model is called."""
        self.instance = instance
        self.store = store
        self.parser = parser
        self.fast_parser = fast_parser
        self.system_one = system_one
        self.tau = tau
        self.use_fast_path = use_fast_path
        self.policy = policy_agent
        self.form_policy = form_policy
        # A question about the timetable ("is Lab 3 free on Friday afternoon?") that came in as a request (the
        # portal, a channel, a replayed history) is answered from the timetable, as the chat answers it, instead of
        # being read as a change: ``lookup(text, request)`` gives (the answer, who read it), or None when the message
        # is not a timetable question. Set by the web app (web.world); without it every message is read as a request.
        self.lookup: Callable[[str, Request], tuple[str, str] | None] | None = None
        # called with each case escalated (after it is logged): the web app answers for a simulated decider
        self.on_escalated: Callable[[Case], None] | None = None
        # called with a case once its change is published (``case.notices`` written): the web app puts each notice
        # in its person's chat (web.world); None: the notices are only on the request
        self.on_published: Callable[[Case], None] | None = None
        self.negotiator = negotiator
        self.responders = dict(responders or {})
        self.approvers = set(approvers)
        self.semester = semester
        self.swap_reader = swap_reader or RuleSwapReader(instance)
        self.extra_reader = ExtraReader(instance)
        self.offer_choices = offer_choices
        self.current_week = current_week
        self.today = today or date.today
        self.club_desk = club_desk
        self.cases: dict[str, Case] = {}
        # held from adding a request's constraints until its proposal is made (a negotiation included): the
        # next request then builds on that proposal. Reading, the policy check and approval go on meanwhile.
        self._solving = threading.RLock()

    # -- setup -------------------------------------------------------------------

    def bootstrap(self, constraints: Iterable[Constraint], assignment=None, approved_by: str = "bootstrap") -> None:
        """Load the constraints in force and publish the starting timetable."""
        constraints = list(constraints)
        self.store.add_constraints(constraints)
        if assignment is None:
            result = TimetableSolver(self.instance, constraints, time_limit=60).solve()
            if not result.ok:
                raise RuntimeError(f"cannot build the starting timetable: {result.status}")
            assignment = result.assignment
        v = self.store.propose_version(assignment, case_id="bootstrap")
        self.store.publish_version(v.version, approved_by)
        self.store.log(None, "bootstrap", version=v.version)

    @contextlib.contextmanager
    def released(self):
        """While waiting for a person (a choice of times, a colleague's answer in a negotiation): let other
        requests solve meanwhile instead of holding them up for the reply deadline. What this request then
        proposes was built on the timetable as it stood, and is rebased onto the newer one when approved
        (``approve``), as any proposal is."""
        lock = self._solving
        if not lock._is_owned():
            yield
            return
        state = lock._release_save()
        try:
            yield
        finally:
            lock._acquire_restore(state)

    # -- the lifecycle ---------------------------------------------------------------

    def _advance(self, case: Case, status: RequestStatus, **info) -> None:
        case.request.advance(status)
        self.store.save_request(case.request)
        self.store.log(case.id, status.value, **info)
        if status == S.ESCALATED and self.on_escalated is not None:
            self.on_escalated(case)  # the web app's simulated deciders (the exam cell's autopilot)

    def _parse(self, case: Case) -> ParseResult:
        r = case.read_as or case.request
        if case.form is not None:  # made with clicks: the intents are given, nothing to read or route
            case.route = "form+intents"
            base = self.store.current_version()
            week = self.current_week() if self.current_week else None
            case.compiled = compile_intents(self.instance, r, case.form,
                                            timetable=_regular(base.assignment) if base else {}, current_week=week)
            return case.compiled.result
        if self.system_one is not None:
            d = self.system_one.decide(r)
            # plain Python values: numpy scalars from System One cannot be sent as JSON
            sure = bool(d.fast_path(self.tau)) and self.fast_parser is not None
            fast = sure and (self.use_fast_path is None or self.use_fast_path())
            case.routing = {"request_type": d.request_type.value, "type_p": round(float(d.type_p), 4),
                            "action": d.action.value, "action_p": round(float(d.action_p), 4),
                            "tau": float(self.tau), "fast_path": fast, "latency_ms": round(float(d.latency_ms), 2)}
            if sure and not fast:
                case.routing["fast_path_skipped"] = True  # sure, but the model reads it
            if fast:
                case.route = "fast"
                result = self._read(case, self.fast_parser)
                # System One is sure of the kind of request, not of the reading: one the fast parser could not
                # settle (a question) goes to System Two rather than back to the sender
                if result.action != ExpectedAction.CLARIFY:
                    return result
                case.compiled = None
                case.routing["fast_path_deferred"] = True
        case.route = "system_two"
        return self._read(case, self.parser)

    def _outside_hours(self, case: Case) -> str | None:
        """A class asked for outside teaching hours or at the weekend ("Thursday at 6 pm", "Saturday
        morning"): not a time the timetable has, but one the head of department may approve (P-HOURS). From
        the message itself, so a reader that drops the time it cannot place ("6 pm") does not hide it; or from
        the reader's own question about it ("No class starts at 6 pm"). Returned: what was asked for."""
        if case.request.role not in (Role.FACULTY, Role.HOD, Role.GUEST_FACULTY):
            return None
        text = (case.read_as or case.request).raw_text
        if said := _asks_outside_hours(text, self.instance):
            return said
        q = case.compiled.question if case.compiled is not None else None
        out = case.compiled.output if case.compiled is not None else None
        if not q or out is None or not (q.startswith("No class starts") or q.startswith("There are no classes on")):
            return None
        return q if any(i.kind in ("move", "extra_class", "only") for i in out.intents) else None

    def _to_head(self, case: Case, asked: str) -> Case:
        """Sent to the head of department, who may approve a class outside hours (P-HOURS): granted, the
        timetable office arranges the room and tells the students (``decide``)."""
        heads = [f.name for f in self.instance.faculty if f.role == Role.HOD]
        brief = (f"{self._name(case.request.sender_id)} asks for a class outside teaching hours or at the weekend: "
                 f"\"{case.request.raw_text.strip()}\". The timetable has no such time ({asked.split(';')[0]}), so it "
                 "is held outside it if approved; the timetable office then books a room and tells the students.")
        case.outcome = Outcome(status="escalated", step=0, escalation=EscalationBrief(
            to="hod", reason="outside teaching hours", text=brief, rule="P-HOURS", asked=asked.split(";")[0],
            grant=("The class is held at the time asked for, outside the weekly timetable: the timetable office books "
                   "a room for it and tells the students. Nothing else in the timetable is re-solved.")))
        case.routing["outside_hours"] = True
        if case.status == S.CLASSIFIED:
            self._advance(case, S.POLICY_CHECKED, verdict="needs_approval", cited=["P-HOURS"])
        case.reply = ("Classes outside 9 am to 5 pm, or at the weekend, need the prior approval of the Head of "
                      f"Department (\u00a72.1 Working hours). Your request has been sent to "
                      f"{heads[0] if heads else 'the Head of Department'}; you will hear back here.")
        self._advance(case, S.ESCALATED, to="hod", reason="outside teaching hours", cited=["P-HOURS"], brief=brief)
        return case

    def _name(self, person: str) -> str:
        f = self.instance.faculty_by_id.get(person)
        return f.name if f else person

    def _closes_own_rooms(self, case: Case) -> bool:
        """A room closure ("Lab 2 is unavailable in week 11", "Room 103 is needed for exams") from the exam cell
        or the lab in-charge, the people the rules make responsible for room availability."""
        cons = case.parse.constraints if case.parse else []
        return (case.request.role in (Role.EXAM_CELL, Role.LAB_INCHARGE) and bool(cons)
                and all(c.scope.room and c.type in (ConstraintType.UNAVAILABLE, ConstraintType.AVOID) for c in cons))

    def _where_now(self, case: Case) -> list[tuple[str, int]]:
        """(day, clock hour) of each class the reading's constraints pick out, in the published timetable."""
        base = self.store.current_version()
        if base is None or case.parse is None:
            return []
        out = []
        for c in case.parse.constraints or []:
            for sid in [c.scope.session, *(c.sessions or [])]:
                if sid and (p := base.assignment.get(sid)) is not None:
                    out.append((p.day, int(hour(p.slot).split()[0]) % 12 + (12 if hour(p.slot).endswith("pm") else 0)))
        return out

    def _room_report(self, case: Case) -> str | None:
        """The rooms a refused request was only about ("Room 101"), when its sender may not change a room's
        availability (a teacher, a student): what they wrote is a report for the office. None otherwise."""
        out = case.parse.output if case.parse else None
        if out is None or not out.constraints or case.request.role not in (Role.FACULTY, Role.GUEST_FACULTY, Role.STUDENT):
            return None
        if any(c.scope_kind != "room" or c.type not in ("unavailable", "avoid") for c in out.constraints):
            return None
        rooms = self.instance.room_by_id
        names = list(dict.fromkeys(rooms[c.scope_id].name if c.scope_id in rooms else c.scope_id
                                   for c in out.constraints))
        return " and ".join(names)

    def _reader(self, case: Case) -> str:
        """Who read the request: "form" (intents given), "rules" (the rule parser: the fast path, or the stand-in
        while the model is away) or "model". A parser that does not say counts as the model unless it is the
        rule parser itself (offline)."""
        if case.form is not None:
            return "form"
        if case.route.startswith("fast"):
            return "rules"
        said = getattr(self.parser, "reader", None)
        if callable(said) and (who := said()):
            return who
        return "rules" if isinstance(self.parser, RuleParser) else "model"

    def _read(self, case: Case, parser) -> ParseResult:
        """Intents resolved against the published timetable when ``parser`` reads them, else the plain parse."""
        compile_ = getattr(parser, "compile", None)
        if compile_ is not None:
            base = self.store.current_version()
            week = self.current_week() if self.current_week else None
            case.compiled = compile_(case.read_as or case.request, _regular(base.assignment) if base else {}, week,
                                     today=self.today())
            if case.compiled is not None:
                case.route += "+intents"
                return case.compiled.result
        return parser.parse(case.read_as or case.request)

    def _policy_for(self, case: Case):
        """The policy check for ``case``: the rule-based one for a request made with clicks, when given."""
        return self.form_policy if case.form is not None and self.form_policy is not None else self.policy

    def submit(self, request: Request, form: IntentOutput | None = None) -> Case:
        """Take a request through the pipeline. ``form``: the request made with clicks (cancel, move or
        swap a class), its intents given instead of read from ``request.raw_text``, which is then only the
        request in words for the people who see it. The sender is told what of their message the reading
        did not use (``Case.unused``), whatever became of the rest."""
        case = self._submit(request, form)
        if case.unused:
            case.reply = (f"{case.reply}\n\nNot used from your message: {' '.join(case.unused)} If it matters, "
                          "withdraw this and send it again in other words.")
        return case

    def _submit(self, request: Request, form: IntentOutput | None) -> Case:
        case = Case(request=request, form=form)
        self.cases[case.id] = case
        if self.store.request(request.id) is None:
            self.store.add_request(request)
        self.store.log(case.id, "received", channel=request.channel.value, sender=request.sender_id)

        flagged = injected(request.raw_text)
        if flagged:
            # instructions to the system are never read; a message that is nothing else is refused unread
            rest = request_part(request.raw_text)
            if rest is None:
                return self._refuse_unread(case)
            case.read_as = request.model_copy(update={"raw_text": rest})
        if form is None and (spelled := expand_shorthand(self.instance, (case.read_as or request).raw_text)):
            # "pls shift my DBMS lec": read as "please shift my Database Systems lecture"
            case.read_as = (case.read_as or request).model_copy(update={"raw_text": spelled})
            self.store.log(case.id, "spelled_out", text=spelled)
        if form is None and self.lookup is not None and (asked := self._timetable_question(case)) is not None:
            return asked
        t = time.perf_counter()
        case.parse = self._parse(case)
        case.timings["parse"] = time.perf_counter() - t
        out = case.compiled.output if case.compiled is not None else None
        if out is not None and out.action == "compile":
            case.understood = [say(self.instance, i, request.sender_id) for i in out.intents]
            if form is None:
                case.unused = unaccounted((case.read_as or request).raw_text, out, self._where_now(case))
                if case.unused:
                    self.store.log(case.id, "unused", notes=case.unused)
        case.routing["reader"] = self._reader(case)
        if flagged:
            case.routing["injection"] = True
        self._advance(case, S.CLASSIFIED, route=case.route, action=case.parse.action.value,
                      routing=case.routing, seconds=round(case.timings["parse"], 3))
        if self._class_rep_report(case):
            return self._report(case)
        if (colleague := self._orders_colleague_class(case)) is not None:
            return self._refuse_colleague(case, colleague)
        if (asked := self._outside_hours(case)) is not None:
            return self._to_head(case, asked)
        if (parts := self._parts(case)) is not None:
            return self._split(case, parts)
        action = case.parse.action
        if (ruled := self._exam_weeks(case)) is not None:
            return ruled

        if case.form is None and (amended := self._amend(case)) is not None:
            return amended
        if action == ExpectedAction.WITHDRAW:
            return self._withdraw_by_message(case)
        if case.intents(ClubIntent):
            return self._club(case)
        if self._is_swap(case):
            return self._swap(case)
        if self._is_extra(case):
            return self._extra(case)
        if case.intents(CancelIntent) and action != ExpectedAction.REFUSE:
            return self._cancel(case)
        if action == ExpectedAction.REFUSE:
            if rooms := self._room_report(case):
                # "the AC in Room 101 is not working": a report about a room, not a change its sender may make
                case.reply = (f"Thank you. Your report about {rooms} has been passed to the timetable office, "
                              "which can close the room or move classes out of it.")
                self._advance(case, S.FORWARDED, to="coordinator", report="room")
                return case
            case.reply = f"Your request was not processed: {case.parse.refusal}. The timetable coordinator has been informed."
            self._advance(case, S.REFUSED, reason=case.parse.refusal)
            return case
        if action == ExpectedAction.ANSWER:
            if self.policy is None:
                case.reply = "Your question has been passed to the timetable coordinator."
                self._advance(case, S.FORWARDED, to="coordinator")
                return case
            case.policy = self.policy.review(request.raw_text, case.parse)
            case.reply = case.policy.answer or case.policy.explanation
            self._advance(case, S.ANSWERED, cited=case.policy.cited)
            return case
        if action == ExpectedAction.INVESTIGATE and case.compiled is not None and case.compiled.clashes:
            return self._clash(case)
        if action in (ExpectedAction.INVESTIGATE, ExpectedAction.OUT_OF_SCOPE):
            to = "coordinator" if action == ExpectedAction.INVESTIGATE else "the relevant office"
            case.reply = f"Thank you. Your message has been passed to {to}."
            self._advance(case, S.FORWARDED, to=to)
            return case

        # compile or clarify: check policy first
        if (policy := self._policy_for(case)) is not None and action == ExpectedAction.COMPILE:
            t = time.perf_counter()
            case.policy = policy.review(request.raw_text, case.parse)
            case.timings["policy"] = time.perf_counter() - t
            if case.policy.verdict == Verdict.FORBIDDEN and self._closes_own_rooms(case):
                # the exam cell or lab in-charge saying a room is taken: theirs to say (P-AUTHORITY); a reading of
                # the rules that forbids it ("week 8 is for exams") is the check misreading what is asked
                case.policy.overruled = (f"room availability is reported by the {request.role.value.replace('_', ' ')}"
                                         f" (P-AUTHORITY), not decided by the rules: {case.policy.explanation}")
                case.policy.verdict = Verdict.ALLOWED
        self._advance(case, S.POLICY_CHECKED,
                      verdict=case.policy.verdict.value if case.policy else "not_checked")
        if action == ExpectedAction.COMPILE and (passed := self._passed(case.parse.constraints)):
            case.reply = passed
            self._advance(case, S.CLARIFICATION, missing=["weeks"])
            return case
        if action == ExpectedAction.CLARIFY:
            out = case.parse.output
            case.reply = (out.clarifying_question if out and out.clarifying_question else
                          "Could you give the days, times and weeks you mean?")
            self._advance(case, S.CLARIFICATION, missing=out.missing if out else [])
            return case
        if case.policy and case.policy.verdict == Verdict.FORBIDDEN:
            case.reply = (f"Your request cannot be granted: {case.policy.explanation}"
                          + (f" Alternative: {case.policy.alternative}" if case.policy.alternative else ""))
            self._advance(case, S.DENIED, cited=case.policy.cited)
            return case
        if case.policy and case.policy.verdict == Verdict.NEEDS_APPROVAL:
            return self._needs_approval(case)

        with self._solving:
            case.constraints = (case.parse.constraints if case.compiled is not None else
                                self._on_named_day(case, case.parse.constraints))
            case.superseded = self._supersede(case)
            self.store.add_constraints(case.constraints, request_id=case.id)
            self._advance(case, S.COMPILED, constraints=[c.id for c in case.constraints],
                          obligations=case.policy.obligations if case.policy else [], superseded=case.superseded)
            return self._solve(case)

    def _needs_approval(self, case: Case) -> Case:
        """Sent up because the policy check found the request needs approval: to whoever the cited rule says
        approves it (``_approver``), as an escalation they can grant (it is then solved, or an extra class placed,
        and goes to the office's approval) or decline. Before, it was sent up to nobody."""
        r, cited = case.request, case.policy.cited
        to, by = self._approver(cited)
        brief = (f"{self._name(r.sender_id)} asks: \"{r.raw_text.strip()}\". The policy check found it needs "
                 f"approval{' under ' + ', '.join(cited) if cited else ''}: {case.policy.explanation}")
        case.outcome = Outcome(status="escalated", step=0, escalation=EscalationBrief(
            to=to, reason="needs approval", text=brief, rule=cited[0] if cited else "needs approval",
            asked=r.raw_text.strip(),
            grant="The request is solved like any other change and then goes to the timetable office for approval."))
        case.reply = f"Your request needs approval and has been sent to {by}: {case.policy.explanation}"
        self._advance(case, S.ESCALATED, to=to, reason="needs approval", cited=cited, brief=brief)
        return case

    def _approver(self, cited: list[str]) -> tuple[str, str]:
        """Who approves what the cited rules need approval for, as the rule says it ("the prior approval of the Head
        of Department"), and in words: hod, exam_cell (the Controller of Examinations), dean, or the timetable
        office (the coordinator, also for a rule that names nobody)."""
        rules = self._rules()
        for rid in cited:
            text = rules.get(rid, "")
            for pattern, to, by in ((r"head of (?:the )?department|\bhod\b", "hod", "the Head of Department"),
                                    (r"controller of examinations|examination cell", EXAM_CELL, "the examination cell"),
                                    (r"\bdean\b", "dean", "the Dean")):
                if re.search(pattern, text, re.IGNORECASE):
                    return to, by
        return "coordinator", "the timetable office"

    def _rules(self) -> dict[str, str]:
        """The rules in force (id -> text), from the policy check (or, behind the web app's fallback, its
        rule-based stand-in, which is given the same corpus)."""
        for p in (self.policy, getattr(self.policy, "backup", None)):
            if isinstance(rules := getattr(p, "rules", None), dict):
                return {rid: r.text for rid, r in rules.items()}
        return {}

    def _exam_weeks(self, case: Case) -> Case | None:
        """A teacher's change in the weeks around the examinations, decided by the examination regulations
        themselves (``pipeline.exams``): refused in an examination week, escalated to the examination cell (the Controller's approval) in
        the weeks next to one. None: not a change, no week near an examination, or the regulations are not loaded."""
        c = case.compiled
        refs = [*c.swaps, *c.extras, *c.cancels] if c is not None else []
        if (case.request.role not in TEACHERS or case.form is not None
                or (case.parse.action != ExpectedAction.COMPILE and not refs)):
            return None
        weeks = [w for x in case.parse.constraints for w in (x.when.weeks or [])]
        weeks += [w for r in refs for w in (getattr(r, "weeks", None) or [])]
        ruling = exam_ruling(weeks, self._rules(), extra=bool(c is not None and c.extras))
        if ruling is None:
            return None
        case.route += "+exams"
        case.reply = ruling.message
        refused = ruling.action == "refuse"
        case.policy = PolicyDecision(Verdict.FORBIDDEN if refused else Verdict.NEEDS_APPROVAL, cited=ruling.cited,
                                     explanation=ruling.message)
        self._advance(case, S.POLICY_CHECKED, verdict=case.policy.verdict.value, exams=ruling.weeks)
        if refused:
            self._advance(case, S.DENIED, cited=ruling.cited, reason=ruling.reason)
        else:  # sent to the examination cell (role exam_cell) to decide, as an escalation it can grant or decline:
            # granted, the request is compiled and solved and goes to the office's approval like any change
            brief = (f"{self._name(case.request.sender_id)} asks for a change in {_weeks_text(ruling.weeks)}, next to "
                     f"the examinations: \"{case.request.raw_text.strip()}\". Under {', '.join(ruling.cited)} no change "
                     "may take effect in the weeks around the examinations without the Controller of Examinations' "
                     "written approval: grant it if no examination is affected.")
            case.outcome = Outcome(status="escalated", step=0, escalation=EscalationBrief(
                to=EXAM_CELL, reason=ruling.reason, text=brief, rule=ruling.cited[0] if ruling.cited else None,
                asked=case.request.raw_text.strip(),
                grant="The request is solved like any other change and then goes to the timetable office for approval."))
            self._advance(case, S.ESCALATED, to=EXAM_CELL, cited=ruling.cited, reason=ruling.reason, brief=brief)
        return case

    def _passed(self, constraints: Iterable[Constraint]) -> str | None:
        """The question to ask when a request names a week that is over (the form refuses the same)."""
        now = self.current_week() if self.current_week else None
        weeks = sorted({w for c in constraints for w in (c.when.weeks or [])})
        if now is None or not weeks or weeks[0] >= now:
            return None
        return f"Week {weeks[0]} has passed; it is week {now} now. Which weeks do you mean?"

    def _timetable_question(self, case: Case) -> Case | None:
        """A question about the timetable, answered from it (``self.lookup``): the case ends Answered with the
        answer as its reply, and nothing is parsed, solved or passed on. None: not a timetable question, or the
        lookup failed; the message is then read as a request, as before."""
        text = (case.read_as or case.request).raw_text
        try:
            got = self.lookup(text, case.request) if self.lookup else None
        except Exception as e:  # noqa: BLE001 - a failed lookup must not lose the request
            self.store.log(case.id, "lookup_failed", error=f"{type(e).__name__}: {e}")
            return None
        if got is None:
            return None
        answer, reader = got
        case.parse = ParseResult(request=case.request, output=None, refusal=None)
        case.route, case.routing = "lookup", {"reader": reader}
        self._advance(case, S.CLASSIFIED, route=case.route, action="answer", routing=case.routing)
        case.reply = answer
        self._advance(case, S.ANSWERED, timetable_question=True)
        return case

    def _refuse_unread(self, case: Case) -> Case:
        why = "it gives the system instructions rather than asking for a change"
        case.parse = ParseResult(request=case.request, output=None, refusal=why)
        case.route, case.routing = "refused", {"reader": "none", "injection": True}
        self._advance(case, S.CLASSIFIED, route=case.route, action="refuse", routing=case.routing)
        case.reply = f"Your request was not processed: {why}. The timetable coordinator has been informed."
        self._advance(case, S.REFUSED, reason=why)
        return case

    # -- a message of several parts ---------------------------------------------------------

    def _parts(self, case: Case) -> list[list] | None:
        """How a message that asks for several things is split, or None to handle it whole. What becomes
        constraints and resolves stays together (one change, solved and negotiated as one); so do its
        cancellations ("cancel my classes on Tuesdays and Thursdays in weeks 11-12": one request, one approval,
        not one a day); every other part is a request of its own: a swap (consent), an extra class, a
        booking, and any part that cannot be resolved (its own question), so one unclear part no longer
        holds up the rest, and none is dropped."""
        c = case.compiled
        if case.form is not None or c is None or c.output is None or c.output.action != "compile":
            return None
        intents = c.output.intents
        if len(intents) < 2:
            return None
        if any(isinstance(i, (AnyOf, Ranked, FallbackIntent)) or getattr(getattr(i, "which", None), "same", None)
               or getattr(getattr(i, "other", None), "same", None) for i in intents):
            return None  # a program: its items refer to each other, so it is read and solved whole
        base = self.store.current_version()
        timetable = _regular(base.assignment) if base else {}
        week = self.current_week() if self.current_week else None
        together, cancels, apart = [], [], []
        unclear = 0
        unclear_ones = []
        for i in intents:
            alone = compile_intents(self.instance, case.read_as or case.request,
                                    IntentOutput(action="compile", intents=[i]), timetable=timetable, current_week=week)
            if alone.question is not None:
                apart.append(i)  # unclear: its own question
                unclear += 1
                unclear_ones.append(i)
            elif isinstance(i, CONSTRAINT_INTENTS):
                together.append(i)
            elif isinstance(i, CancelIntent):
                cancels.append(i)  # every cancellation of the message: one request for the office to approve
            else:
                apart.append(i)
        if unclear == len(intents):
            # nothing could go ahead on its own: splitting would only turn one question into several, each
            # about a fragment of the message, so the message is asked about whole
            return None
        about = [self._about(case, i, timetable) for i in intents]
        if any(about[n] & about[m] for n, i in enumerate(intents) if i in unclear_ones
               for m in range(len(intents)) if m != n):
            # an unclear part is about a class another part is also about ("my practical needs the routers, and
            # it has to be on Tuesday afternoon", from someone who teaches it to two sections): split, the other
            # part would go ahead on its own and the answer would reach only the unclear one, so the message is
            # asked about whole and the answer is read into every part
            return None
        parts = ([together] if together else []) + ([cancels] if cancels else []) + [[i] for i in apart]
        return parts if len(parts) > 1 else None

    def _about(self, case: Case, intent, timetable: Mapping[str, Placement]) -> set[str]:
        """The classes a part of a message is about, every one its reference fits (none when it names none)."""
        ref = getattr(intent, "which", None) or getattr(intent, "mine", None)
        if ref is None:
            return set()
        try:
            return set(classes(self.instance, ref, case.request.sender_id, timetable))
        except Unresolved:
            return set()

    def _split(self, case: Case, parts: list[list]) -> Case:
        """Each part goes through the pipeline as a request of its own, its intents as read and its words
        written from them (what is read again when its sender answers a question about it)."""
        sender = case.request.sender_id
        children = []
        for n, group in enumerate(parts, 1):
            words = " ".join(say(self.instance, i, sender) for i in group)
            child = case.request.model_copy(update={"id": f"{case.id}-{n}", "raw_text": words,
                                                    "status": S.RECEIVED})
            # a clash report is looked into (forwarded with the clash, ``_clash``), not solved as an empty change
            action = "investigate" if all(isinstance(i, ClashReport) for i in group) else "compile"
            children.append(self.submit(child, form=IntentOutput(action=action, intents=group)))
            children[-1].routing["part_of"] = case.id
        case.parts = [x.id for x in children]
        case.route += "+split"
        case.reply = (f"Your message asks for {len(children)} things, so each is handled as a request of its own:\n"
                      + "\n".join(f"- {x.id} ({x.request.raw_text}) {x.reply}" for x in children))
        self._advance(case, S.ANSWERED, parts=case.parts)
        return case

    # -- swaps (P-SWAP) ------------------------------------------------------------------

    def _is_swap(self, case: Case) -> bool:
        """The parser says swap, or (for parsers never trained on swaps) the
        message has a swap word and names a colleague. Students keep their
        refusal; questions about the swap rule stay questions."""
        if case.request.role == Role.STUDENT or case.parse.action in (
                ExpectedAction.ANSWER, ExpectedAction.INVESTIGATE, ExpectedAction.OUT_OF_SCOPE):
            return False
        if case.compiled is not None:
            return bool(case.intents(SwapIntent)) and case.parse.action != ExpectedAction.REFUSE
        return case.parse.request_type == RequestType.SWAP or looks_like_swap(self.instance, case.request)

    def _swap_reading(self, case: Case, timetable: Mapping[str, Placement]) -> SwapReading:
        """Which two classes: from the parsed intents when there are any, else the swap reader."""
        c, r = case.compiled, case.request
        if c is None:
            return self.swap_reader.read(r, timetable)
        if c.question is not None:
            return SwapReading(question=c.question, missing=c.missing)
        ref = c.swaps[0]
        return check_swap(self.instance, r, ref.mine, ref.theirs, timetable, ref.weeks)

    def _swap(self, case: Case) -> Case:
        r = case.request
        case.route += "+swap"
        if case.compiled is not None:
            weeks_hint = case.compiled.swaps[0].weeks if case.compiled.swaps else None
        else:
            weeks_hint = RuleParser(self.instance).weeks(r.raw_text)
        base = (self.store.current_version(weeks_hint[0]) if weeks_hint else None) or self.store.current_version()
        reading = self._swap_reading(case, base.assignment if base else {})
        if reading.refusal:
            case.reply = f"Your request was not processed: {reading.refusal}."
            self._advance(case, S.REFUSED, reason=reading.refusal)
            return case
        plan = reading.plan
        pins = swap_constraints(self.instance, plan, r) if plan else []
        if errors := [e for c in pins for e in validate_constraint(c, self.instance)]:
            plan, pins = None, []
            reading.question, reading.missing = f"That swap cannot be written as a timetable change ({errors[0]}).", ["session"]
        if plan and (policy := self._policy_for(case)) is not None:
            t = time.perf_counter()
            view = ParseResult(request=r, output=case.parse.output, constraints=pins)
            case.policy = policy.review(r.raw_text, view)
            case.timings["policy"] = time.perf_counter() - t
        self._advance(case, S.POLICY_CHECKED, verdict=case.policy.verdict.value if case.policy else "not_checked",
                      swap=_plan_view(plan) if plan else None)
        if plan is None:
            case.reply = reading.question or "Which two classes would you like to swap?"
            self._advance(case, S.CLARIFICATION, missing=reading.missing)
            return case
        rule = next((f"§{x.number} {x.title}" for x in getattr(self.policy, "rules", {}).values()
                     if x.id == "P-SWAP"), None) if self.policy else None
        # P-SWAP itself asks for consent and for the coordinator to be informed: that
        # is this step and the approval below. A verdict under any other rule stands.
        other_rules = set(case.policy.cited) - {"P-SWAP"} if case.policy else set()
        if case.policy and case.policy.verdict == Verdict.FORBIDDEN and other_rules:
            case.reply = (f"Your request cannot be granted: {case.policy.explanation}"
                          + (f" Alternative: {case.policy.alternative}" if case.policy.alternative else ""))
            self._advance(case, S.DENIED, cited=case.policy.cited)
            return case
        if case.policy and case.policy.verdict != Verdict.ALLOWED and other_rules:
            case.reply = f"Your request needs approval and has been sent up: {case.policy.explanation}"
            self._advance(case, S.ESCALATED, reason="needs approval", cited=case.policy.cited)
            return case

        responder = self.responders.get(plan.counterpart)
        if responder is None:
            case.reply = "Your colleague cannot be reached through the system; the coordinator has been informed."
            self._advance(case, S.ESCALATED, to="coordinator", reason=f"no inbox for {plan.counterpart}")
            return case
        message = consent_message(self.instance, plan, rule)
        self.store.log(case.id, "swap_consent_requested", to=plan.counterpart, message=message.model_dump())
        reply = responder.respond(message)
        if reply.decision is None and self.negotiator.reply_parser is not None:
            reply = self.negotiator.reply_parser.parse(message, reply.text)
        agreed = reply.decision == "accept" and (reply.choice or "A").strip().upper() == "A"
        self.store.log(case.id, "swap_consent", by=plan.counterpart, agreed=agreed,
                       reply=reply.model_dump(exclude_none=True))
        if not agreed:
            who = self.instance.faculty_by_id[plan.counterpart].name
            why = "did not reply in time" if reply.decision == "no_reply" else "did not agree"
            case.reply = (f"{who} {why}, so the swap was not made. A swap needs the consent of both of you."
                          + self._instead_of_swap(plan))
            self._advance(case, S.DENIED, reason=f"swap declined by {plan.counterpart}", cited=["P-SWAP"])
            return case

        with self._solving:
            case.constraints = pins
            superseded = self._supersede(case)
            self.store.add_constraints(case.constraints, request_id=case.id)
            self._advance(case, S.COMPILED, constraints=[c.id for c in pins], superseded=superseded,
                          consent=plan.counterpart)
            return self._solve(case)

    def _instead_of_swap(self, plan) -> str:
        """After a declined swap: the free hours the sender's class could move to on its own in that week, so
        nobody else has to agree (none listed when the week has none, or the swap was for no week)."""
        week = plan.weeks[0] if plan.weeks else None
        if week is None:
            return ""
        session = self.instance.session_by_id.get(plan.mine)
        base = self._tip(week, follow_semester=False)
        if session is None or base is None:
            return ""
        cancelled, _ = self._cancelled(week)
        free = free_placements(self.instance, session, week, _regular(base.assignment), self.store.constraints(),
                               self.store.extras(week), skip=cancelled)
        free = [p for p in free if (p.day, p.slot) != (plan.mine_at.day, plan.mine_at.slot)][:3]
        if not free:
            return ""
        times = "; ".join(placement_text(self.instance, p) for p in free)
        return (f" Instead, {session_name(self.instance, plan.mine)} could move on its own in week {week}, with "
                f"nobody else's class changing: {times}. Ask to move it to one of these if that suits you.")

    # -- cancellations and withdrawals in a message (intent parsers) -------------------------

    def _cancel(self, case: Case) -> Case:
        """A class not held in some weeks. Without make-up classes (the form's box left empty) it is done
        like an absence: the class is left out of those weeks' timetables, and the change waits for the
        office's approval like any other. With make-up classes asked for, the coordinator arranges them, so
        the request goes to them with the class and weeks resolved. Unclear: back to the sender."""
        c = case.compiled
        case.route += "+cancel"
        if c.question is not None:
            self._advance(case, S.POLICY_CHECKED, verdict="not_checked")
            case.reply = c.question
            self._advance(case, S.CLARIFICATION, missing=c.missing)
            return case
        base = self.store.current_version()
        placed = base.assignment if base else {}
        if case.request.makeup is False and c.cancels and all(x.weeks and x.session in placed for x in c.cancels):
            n = case.id.split("-")[-1]
            cons = [Constraint(id=f"C-{n}-{i}", type=ConstraintType.UNAVAILABLE, hard=True,
                               tier=Tier.VERIFIED_UNAVAILABILITY, owner=case.request.sender_id,
                               scope=Scope(session=x.session), when=When(weeks=list(x.weeks)),
                               source=Source(request=case.id))
                    for i, x in enumerate(c.cancels, 1)]
            self._advance(case, S.POLICY_CHECKED, verdict="not_checked")
            with self._solving:
                case.constraints = cons
                self.store.add_constraints(cons, request_id=case.id)
                self._advance(case, S.COMPILED, constraints=[k.id for k in cons], obligations=[], superseded=[])
                return self._solve(case)
        what = "; ".join(
            f"{session_name(self.instance, x.session)}"
            + (f" on {placement_text(self.instance, placed[x.session])}" if x.session in placed else "")
            + (f" in week {x.weeks[0]}" if x.weeks and len(x.weeks) == 1 else
               f" in weeks {x.weeks[0]}-{x.weeks[-1]}" if x.weeks else "")
            for x in c.cancels)
        case.reply = (f"Your request to cancel {what} has been passed to the timetable coordinator, who arranges "
                      "cancellations and make-up classes.")
        self._advance(case, S.FORWARDED, to="coordinator",
                      cancel=[{"session": x.session, "weeks": x.weeks} for x in c.cancels])
        return case

    def _club(self, case: Case) -> Case:
        """A room for a club activity: the club desk checks it against the semester timetable and the
        bookings, and books it, turns it down with alternatives, or leaves it to the timetable office."""
        c = case.compiled
        case.route += "+club"
        if c.question is not None:
            self._advance(case, S.POLICY_CHECKED, verdict="not_checked")
            case.reply = c.question
            self._advance(case, S.CLARIFICATION, missing=c.missing)
            return case
        ref = c.clubs[0]
        if self.club_desk is None:
            case.reply = "Your room request for the club activity has been passed to the timetable office."
            self._advance(case, S.FORWARDED, to="coordinator", club=ref.__dict__)
            return case
        try:
            reply, outcome = self.club_desk(ref, case.request)
        except ValueError as e:
            case.reply = f"The club desk could not check this ({e}); it has been passed to the timetable office."
            self._advance(case, S.FORWARDED, to="coordinator", club=ref.__dict__)
            return case
        case.reply = reply
        if outcome == "needs_approval":
            self._advance(case, S.FORWARDED, to="coordinator", club=ref.__dict__, outcome=outcome)
        else:
            self._advance(case, S.ANSWERED, club=ref.__dict__, outcome=outcome)
        return case

    def _clash(self, case: Case) -> Case:
        """Students report two classes at once. The published timetable is checked first: a real overlap goes
        to the coordinator with both classes named; none, and the students hear at once where each class is."""
        ref = case.compiled.clashes[0]
        case.route += "+clash"
        base = self.store.current_version()
        placed = _regular(base.assignment) if base else {}
        sessions = [s for s in self.instance.sessions if s.course in ref.courses and s.id in placed]
        if len(ref.courses) < 2 or not sessions:
            case.reply = "Thank you. Your message has been passed to coordinator."
            self._advance(case, S.FORWARDED, to="coordinator")
            return case

        def cells(s) -> set[tuple[str, int]]:
            p = placed[s.id]
            return {(p.day, q) for q in range(p.slot, p.slot + s.duration)
                    if (not ref.days or p.day in ref.days) and (ref.slots is None or q in ref.slots)}

        overlaps = sorted({(a.id, b.id) for a in sessions for b in sessions
                           if a.course < b.course and cells(a) & cells(b)})
        if overlaps:
            pairs = "; ".join(f"{session_name(self.instance, a)} and {session_name(self.instance, b)} on "
                              f"{placement_text(self.instance, placed[a])}" for a, b in overlaps[:3])
            case.reply = (f"We checked the published timetable: {pairs} are at the same time. This has been passed "
                          "to the timetable coordinator to fix.")
            self._advance(case, S.FORWARDED, to="coordinator", clash=[list(x) for x in overlaps])
            return case
        where = "; ".join(f"{session_name(self.instance, s.id)} is on {placement_text(self.instance, placed[s.id])}"
                          for s in sessions[:6])
        case.reply = (f"We checked the published timetable and these do not overlap: {where}. If you still have two "
                      "classes at once (for example an elective the timetable does not list), write again with the "
                      "class names and times and the timetable office will look at it.")
        self._advance(case, S.ANSWERED, clash=[])
        return case

    def _amend(self, case: Case) -> Case | None:
        """"Actually, Wednesday mornings are fine now; only Monday mornings need to stay free": a change to
        the sender's earlier wish, not a new one and not a withdrawal of all of it. The days the message lets
        go of ("Wednesday ... fine", "not Wednesday", "no longer") are taken out of their latest wish that
        names them, or the days it keeps ("only Monday") are put in their place; the earlier wish is
        replaced by the narrower one, which is solved like any request. None: not such a message."""
        text = (case.read_as or case.request).raw_text
        released, kept = _amended_days(text)
        if not released or not _AMENDS.search(text):  # "to Thursday, not Wednesday" is a new request
            return None
        sender = case.request.sender_id
        # the wish that holds a day being let go of ("Wednesday ... fine"): a wish the sender wrote in a request of
        # theirs, never a constraint that came from elsewhere, such as a placement pinned when a negotiation settled
        # ("N2-C-033-P-T", no request: amending it once turned "keep Monday mornings free" into "the practical
        # needs Monday 11 am"), nor any later constraint of theirs that merely names a kept day
        mine = [c for c in self.store.constraints() if c.owner == sender and c.when.days
                and set(c.when.days) & released and c.source.rule is None and c.source.request and not c.members
                and c.type in (ConstraintType.AVOID, ConstraintType.PREFER, ConstraintType.UNAVAILABLE)]
        if not mine:
            return None
        # their latest such wish, by when it was asked (the request's number), not by the id's spelling
        old = max(mine, key=lambda c: (_request_number(c.source.request), c.id))
        days = sorted(kept & set(DAYS) or set(old.when.days) - released, key=DAYS.index)
        if not days or days == list(old.when.days):
            return None  # nothing left (a withdrawal), or nothing changes
        new = old.model_copy(deep=True, update={"id": f"C-{case.id.split('-')[-1]}-1",
                                                "source": old.source.model_copy(update={"request": case.id})})
        new.when = old.when.model_copy(update={"days": days})
        case.route += "+amend"
        case.unused = []
        self._advance(case, S.POLICY_CHECKED, verdict="not_checked", amends=old.source.request)
        with self._solving:
            case.constraints = [new]
            self.store.set_active(old.id, False)
            case.superseded = [old.id]
            self.store.add_constraints(case.constraints, request_id=case.id)
            self._advance(case, S.COMPILED, constraints=[new.id], obligations=[], superseded=case.superseded)
            case = self._solve(case)
        was = ", ".join(FULL_DAY.get(d, d) for d in old.when.days)
        now = ", ".join(FULL_DAY.get(d, d) for d in days)
        case.reply = (f"Your earlier request {old.source.request} now covers {now} only (it was {was}). "
                      f"{case.reply}").strip()
        return case

    def _withdraw_by_message(self, case: Case) -> Case:
        """"Please ignore my earlier request": the sender's one open request is taken back. With several
        open, nothing is withdrawn on a guess: they are listed for the sender to choose. A reading of withdraw
        for a message that says nothing of the kind ("x", "ok") withdraws nothing: it is not understood."""
        sender = case.request.sender_id
        case.route += "+withdraw"
        if not SAYS_WITHDRAW.search((case.read_as or case.request).raw_text):
            case.reply = ("Your message did not say what you need, so nothing was done. Write the change you would "
                          "like, or ask a question about the timetable.")
            self._advance(case, S.ANSWERED, withdrew=None, unclear=True)
            return case
        open_ = [c for c in self.cases.values()
                 if c.request.sender_id == sender and c.id != case.id and self.withdrawable(c) is None]
        if len(open_) == 1:
            taken = self.withdraw(open_[0].id, by=sender)
            case.reply = f"Your request {taken.id} (\"{_short(taken.request.raw_text)}\") was withdrawn. {taken.reply}"
            self._advance(case, S.ANSWERED, withdrew=taken.id)
        elif not open_:
            case.reply = "You have no open request that can still be withdrawn."
            self._advance(case, S.ANSWERED, withdrew=None)
        else:
            listed = "; ".join(f"{c.id} (\"{_short(c.request.raw_text)}\")" for c in open_[-4:])
            case.reply = (f"You have {len(open_)} open requests: {listed}. Withdraw the one you mean from its page, "
                          "or tell us which.")
            self._advance(case, S.ANSWERED, withdrew=None, open=[c.id for c in open_])
        return case

    # -- extra classes ---------------------------------------------------------------------

    def _is_extra(self, case: Case) -> bool:
        """A teacher asks for a class on top of the weekly ones. Questions about extra classes stay
        questions; a refusal (an injection) stays a refusal."""
        if case.compiled is not None:
            return (case.request.role in TEACHERS and bool(case.intents(ExtraIntent))
                    and case.parse.action != ExpectedAction.REFUSE)
        return (case.request.role in TEACHERS and looks_like_extra(case.request.raw_text)
                and case.parse.action not in (ExpectedAction.REFUSE, ExpectedAction.ANSWER, ExpectedAction.OUT_OF_SCOPE))

    def _extra_reading(self, case: Case) -> ExtraReading:
        """Which class, week and times: from the parsed intents when there are any, else the extra reader."""
        c, r = case.compiled, case.request
        if c is None:
            return self.extra_reader.read(r.raw_text, r.sender_id)
        if c.question is not None:
            return ExtraReading(question=c.question, options=["class"] if c.missing == ["session"] else [])
        ref = c.extras[0]
        return ExtraReading(template=self.instance.session_by_id[ref.template], week=ref.weeks[0],
                            days=ref.days, slots=ref.slots)

    def _extra(self, case: Case) -> Case:
        r = case.request
        case.route += "+extra"
        reading = self._extra_reading(case)
        case.extra = {"template": reading.template.id if reading.template else None, "week": reading.week}
        if reading.template is not None and (policy := self._policy_for(case)) is not None:
            t = time.perf_counter()
            case.policy = policy.review(r.raw_text, ParseResult(request=r, output=case.parse.output))
            case.timings["policy"] = time.perf_counter() - t
        self._advance(case, S.POLICY_CHECKED, verdict=case.policy.verdict.value if case.policy else "not_checked",
                      extra=case.extra)
        if reading.question:
            case.reply = reading.question
            self._advance(case, S.CLARIFICATION, missing=["class"] if reading.options else ["week"])
            return case
        if case.policy and case.policy.verdict == Verdict.FORBIDDEN:
            case.reply = (f"Your request cannot be granted: {case.policy.explanation}"
                          + (f" Alternative: {case.policy.alternative}" if case.policy.alternative else ""))
            self._advance(case, S.DENIED, cited=case.policy.cited)
            return case
        if case.policy and case.policy.verdict == Verdict.NEEDS_APPROVAL:
            return self._needs_approval(case)
        return self._place_extra(case)

    @_one_at_a_time
    def _place_extra(self, case: Case) -> Case:
        """Find a time in the week when the teacher, the section and a room are free, and propose the
        week's timetable with the extra class in it. Nothing else moves, so there is nothing to negotiate."""
        r = case.request
        reading = self._extra_reading(case)
        week, label = reading.week, self.extra_reader.label(reading.template)
        session = extra_session(reading.template, case.id)
        # into a free hour, without the solver: nothing of a waiting semester change is made again in it
        base = self._tip(week, follow_semester=False)
        assignment = _regular(base.assignment) if base else {}
        cancelled, _ = self._cancelled(week)
        others = [c for c in self.store.constraints() if c.source.request != case.id]
        extras = [x for x in self.store.extras(week) if x.request != case.id]

        def search(days, slots):
            return free_placements(self.instance, session, week, assignment, others, extras, days, slots, skip=cancelled)

        if case.status != S.COMPILED:
            self._advance(case, S.COMPILED, extra=session.id, template=reading.template.id, week=week)
        found = search(reading.days, reading.slots)
        if not found:
            asked = " ".join(x for x in (
                f"on {' or '.join(FULL_DAY[d] for d in reading.days)}" if reading.days else "",
                f"at {' or '.join(hour(q) for q in reading.slots)}" if reading.slots else "") if x)
            anywhere = search(None, None) if asked else []
            who = "you, the section and a suitable room are" if reading.template.groups else "you and a suitable room are"
            if anywhere:
                case.reply = (f"There is no free hour for the extra {label} class {asked} in week {week}: one when {who} "
                              f"all free. Free times that week: {'; '.join(_spread(anywhere))}. Which would you like?")
            else:
                case.reply = (f"There is no free hour for the extra {label} class anywhere in week {week}: none when "
                              f"{who} all free. Would another week work?")
            self._advance(case, S.CLARIFICATION, missing=["time"], extra=session.id)
            return case

        p = found[0]
        x = ExtraClass(session=session, week=week, placement=p, request=case.id)
        self.store.add_extra(x)
        case.constraints = block_constraints(x, r.sender_id)
        self.store.add_constraints(case.constraints, request_id=case.id)
        case.extra = {"template": reading.template.id, "week": week, "session": session.id, "label": label,
                      "placement": p.model_dump()}
        self._advance(case, S.SOLVED, feasible=True, extra=session.id, placement=p.model_dump())
        teaching = sorted({s.faculty for s in self.instance.sessions})
        gini = round(ConcessionLedger(self.store.ledger()).gini(teaching, self.semester), 4)
        case.fairness = {"gini_before": gini, "gini_after": gini, "conceded": [], "preferences_lost": []}
        self._advance(case, S.FAIRNESS_AUDITED, **case.fairness)
        case.proposal = self.store.propose_version(assignment, case.id, week=week, cancelled=sorted(cancelled),
                                                   parent=base)
        case.reply = (f"An extra {label} class can be held on {when_text(self.instance, p, week)}. Nothing else has "
                      "to move. It is waiting for the coordinator's approval.")
        self._advance(case, S.AWAITING_APPROVAL, version=case.proposal.version, moved=[session.id])
        return case

    def _on_named_day(self, case: Case, constraints: list[Constraint]) -> list[Constraint]:
        """"On Thursday move my Machine Learning class before lunch" means the Machine Learning lecture
        that is on Thursday. A parser names a course's session without seeing the timetable, so a placement
        pinned to a day and times moves to the course's other session of that kind when the one named is
        not on that day and exactly one other is: otherwise it would add a second lecture to the day."""
        sessions = self.instance.session_by_id
        out = []
        for c in constraints:
            s = sessions.get(c.scope.session) if c.scope.session else None
            if s is None or c.type != ConstraintType.PREFER or not c.when.days or not c.when.slots:
                out.append(c)
                continue
            week = c.valid.from_week if c.valid.from_week == c.valid.to_week else None
            base = self.store.current_version(week) or self.store.current_version()
            placed = _regular(base.assignment) if base else {}
            days = set(c.when.days)
            on_day = [o.id for o in self.instance.sessions if o.id != s.id and o.course == s.course
                      and o.kind == s.kind and o.faculty == s.faculty and o.id in placed and placed[o.id].day in days]
            if s.id in placed and placed[s.id].day not in days and len(on_day) == 1:
                self.store.log(case.id, "retargeted", constraint=c.id, named=s.id, to=on_day[0],
                               day=placed[on_day[0]].day)
                c = c.model_copy(update={"scope": c.scope.model_copy(update={"session": on_day[0]})})
            out.append(c)
        return out

    def _supersede(self, case: Case) -> list[str]:
        """A newer request from the same owner replaces their earlier active
        constraint of the same type, tier and scope ("actually, Friday
        instead of Tuesday"), instead of conflicting with it."""
        out = []
        for old in self.store.constraints():
            if old.source.request == case.id or old.source.rule == EXTRA_RULE:  # an extra class is not a wish
                continue
            for c in case.constraints:
                if (old.owner and old.owner == c.owner and old.type == c.type and old.tier == c.tier
                        and old.scope == c.scope and old.when.weeks == c.when.weeks):
                    self.store.set_active(old.id, False)
                    out.append(old.id)
                    break
        return out

    def _solve(self, case: Case) -> Case:
        """Repair the timetable for the request, negotiating where it conflicts. A request for several weeks
        is repaired in each of them (a week has its own timetable): the further weeks are settled first, and
        if any of them escalates, so does the request; their changes are recorded only once it goes through,
        and each becomes a proposal of its own, approved with the first."""
        weeks = sorted({w for c in case.constraints for w in (c.when.weeks or [])})
        week = weeks[0] if weeks else None
        neg = self.negotiator  # one for every case: each negotiation binds its own state

        def on_event(kind: str, data: dict) -> None:
            if kind == "message" and case.status == S.COMPILED:
                self._advance(case, S.SOLVED, feasible=False)
                self._advance(case, S.NEGOTIATING)
            self.store.log(case.id, f"negotiation_{kind}", **data)

        t = time.perf_counter()
        later = []
        for w in weeks[1:]:
            got = self._negotiate_week(w, neg, on_event)
            if got[1].status == "escalated":
                return self._escalate(case, got[1], week=w)
            later.append(got)
        case.timings["later_weeks"] = time.perf_counter() - t
        more_to_do = any(moved or cancelled for _w, _o, _b, _a, moved, cancelled in later)
        base = self._tip(week)
        baseline = _regular(base.assignment) if base else None
        # "already met" only by the published timetable: met by a change still waiting, it waits with it
        live = base is not None and base.version == getattr(
            self.store.current_version(week) or self.store.current_version(), "version", None)
        cancelled, closed = self._cancelled(week)
        if (live and not more_to_do and set(cancelled) == set(base.cancelled)
                and self._already_met(case, base, week)):
            # checked directly, not by the solver: a solve cut short by its time limit could move
            # classes nobody asked to move
            self._advance(case, S.SOLVED, feasible=True, already_met=True)
            self._advance(case, S.FAIRNESS_AUDITED)
            return self._already_fits(case)

        t = time.perf_counter()
        published = self.store.current_version(week) or self.store.current_version()
        outcome = neg.resolve(self.store.constraints(), self.responders, baseline=baseline or {}, week=week,
                              skip=cancelled, ledger=ConcessionLedger(self.store.ledger()), listener=on_event,
                              reference=_regular(published.assignment) if published else None)
        case.timings["solve_and_negotiate"] = time.perf_counter() - t
        case.outcome = outcome
        if case.status == S.COMPILED:
            self._advance(case, S.SOLVED, feasible=outcome.status == "feasible",
                          solver_seconds=round(outcome.result.wall_time, 3) if outcome.result else None)
        if outcome.status != "feasible" and case.status == S.SOLVED:
            self._advance(case, S.NEGOTIATING, rounds=outcome.rounds)
        if outcome.status == "escalated":
            return self._escalate(case, outcome)

        for o in [outcome, *(got[1] for got in later)]:
            self._record(case, o)
        case.fairness = self._audit(outcome, *(got[1] for got in later))
        self._advance(case, S.FAIRNESS_AUDITED, **case.fairness)

        assignment = outcome.result.assignment
        moved = moved_sessions(baseline or {}, assignment)
        chosen = None
        if outcome.status == "feasible" and moved and not cancelled:
            assignment, chosen = self._let_sender_choose(case, assignment, baseline, week, neg, cancelled)
            moved = moved_sessions(baseline or {}, assignment)
        unmet = outcome.notices.get(case.request.sender_id)
        if live and not moved and not cancelled and not unmet and not more_to_do:
            # the timetable already meets it: nothing to publish, but it stays in force
            return self._already_fits(case)
        if live and not moved and not cancelled and not more_to_do:
            # a wish the solver left unmet rather than move other classes: nothing changes, so there is
            # nothing for the coordinator to approve; the wish stays in force for later changes
            wanted = " ".join(n.removeprefix("Your preference could not be fully met: ") for n in unmet)
            case.reply = (f"Nothing was changed. You asked: {wanted} It could not be met without moving other "
                          "classes, so it stays recorded and later changes will try to meet it. If it has to "
                          "happen, withdraw this and send it again as a requirement.")
            self._advance(case, S.PUBLISHED, version=None, moved=[], unmet=True)
            return case
        case.proposal = self.store.propose_version(assignment, case.id, week=week, cancelled=sorted(cancelled),
                                                   parent=base)
        case.more = [self.store.propose_version(a, case.id, week=w, cancelled=sorted(c), parent=b)
                     for w, _o, b, a, m, c in later if m or c]
        case.reply = self._proposal_reply(case, outcome, assignment, moved, week, cancelled, closed, chosen)
        self._advance(case, S.AWAITING_APPROVAL, version=case.proposal.version, moved=moved,
                      **({"more": [v.version for v in case.more]} if case.more else {}))
        return case

    def _already_fits(self, case: Case) -> Case:
        """Nothing moves, but the request stays in force for every later change. A wish is recorded as it is;
        a requirement (a hard constraint) waits for the coordinator, who sees how it was read: a misreading
        recorded unseen ("two classes a day" read as two hours in a row) would bind the timetable from then on."""
        if any(c.hard for c in case.constraints):
            case.reply = ("Your request already fits the timetable, so nothing needs to move. It adds a requirement "
                          "that later changes must keep to, so the timetable office confirms it first.")
            self._advance(case, S.AWAITING_APPROVAL, version=None, moved=[], record_only=True)
            return case
        case.reply = ("Your request already fits the timetable, so nothing needed to move. "
                      "It is recorded, and later changes will keep to it.")
        self._advance(case, S.PUBLISHED, version=None, moved=[])
        return case

    def _negotiate_week(self, week: int, neg: Negotiator, on_event):
        """Settle the request in one further week: (week, outcome, the version it builds on, the repaired
        assignment, the classes it moves, the classes not held that week). Nothing is recorded yet."""
        base = self._tip(week)
        baseline = _regular(base.assignment) if base else {}
        cancelled, _ = self._cancelled(week)
        published = self.store.current_version(week) or self.store.current_version()
        outcome = neg.resolve(self.store.constraints(), self.responders, baseline=baseline, week=week,
                              skip=cancelled, ledger=ConcessionLedger(self.store.ledger()), listener=on_event,
                              reference=_regular(published.assignment) if published else None)
        if outcome.status == "escalated":
            return week, outcome, base, None, [], set()
        assignment = outcome.result.assignment
        return week, outcome, base, assignment, moved_sessions(baseline, assignment), set(cancelled)

    def _escalate(self, case: Case, outcome, week: int | None = None) -> Case:
        """Until someone decides, the request is not in force: later requests must not run into it. A request
        with a fallback ("if that is not possible, cancel it") falls back instead of waiting for a decision."""
        if case.compiled is not None and case.compiled.fallback and not case.routing.get("fell_back"):
            return self._fall_back(case, outcome, week)
        case.outcome = outcome
        if case.status == S.COMPILED:
            self._advance(case, S.SOLVED, feasible=False)
        if case.status == S.SOLVED:
            self._advance(case, S.NEGOTIATING, rounds=outcome.rounds)
        self._withdraw(case)
        case.reply = ("Your request conflicts with others and has been escalated for a decision."
                      + (f" The conflict is in week {week}." if week is not None else ""))
        self._advance(case, S.ESCALATED, to=outcome.escalation.to, reason=outcome.escalation.reason,
                      brief=outcome.escalation.text, **({"week": week} if week is not None else {}))
        return case

    def _fall_back(self, case: Case, outcome, week: int | None) -> Case:
        """What was asked first could only be met by overriding others: it is taken out of force and the
        fallback goes through the pipeline as a request of its own, as a part of a split message does."""
        case.outcome = outcome
        self._withdraw(case)
        sender = case.request.sender_id
        intents = [i.model_copy(update={"id": None}) for i in case.compiled.fallback]
        words = " ".join(say(self.instance, i, sender) for i in intents)
        case.routing["fell_back"] = True
        child = case.request.model_copy(update={"id": f"{case.id}-1", "raw_text": words, "status": S.RECEIVED})
        done = self.submit(child, form=IntentOutput(action="compile", intents=intents))
        done.routing["part_of"] = case.id
        case.parts = [done.id]
        case.reply = ("What you asked first could not be done without overriding others"
                      + (f" in week {week}" if week is not None else "")
                      + f", so your fallback goes ahead instead as {done.id} ({words}) {done.reply}")
        self._advance(case, S.ANSWERED, parts=case.parts, fell_back=True)
        return case

    def _record(self, case: Case, outcome) -> None:
        """What a settled negotiation changed: constraints given up or added, and concessions."""
        for cid in outcome.relaxed:
            self.store.set_active(cid, False)
        known = {c.id for c in self.store.constraints(active_only=False)}
        self.store.add_constraints([c for c in outcome.constraints if c.id not in known], request_id=case.id)
        for e in outcome.concessions:
            self.store.record_concession(e)

    def _proposal_reply(self, case: Case, outcome, assignment: Mapping[str, Placement], moved: list[str],
                        week: int | None, cancelled: set[str], closed: list[str], chosen: str | None) -> str:
        """What the sender of a change now waiting for approval is told; the most specific account first."""
        waiting = "The change is waiting for the coordinator's approval."
        weeks = [week, *(v.week for v in case.more)]
        if len(weeks) > 1:
            waiting = (f"It covers {_weeks_text(weeks)}, each with its own change; they are waiting for the "
                       "coordinator's approval.")
        others = len(set(moved) - cancelled)
        also = f"; {others} other {'class moves' if others == 1 else 'classes move'}" if others else ""
        if "+cancel" in case.route and (mine := self._absent(week, only=case.id)):
            names = " and ".join(session_name(self.instance, sid) for sid in sorted(mine))
            return (f"Recorded: {names} {'is' if len(mine) == 1 else 'are'} cancelled in {_weeks_text(weeks)}, with "
                    "no make-up class, as you asked" + also + ". Your sections are told once the timetable office "
                    "approves the change.")
        if mine := self._absent(week, only=case.id):
            n = len(mine)
            return (f"Recorded: you are away in {_weeks_text(weeks)}. Your {n} {'class' if n == 1 else 'classes'} "
                    f"{'is' if n == 1 else 'are'} cancelled {'that week' if len(weeks) == 1 else 'each of those weeks'}"
                    ", with no make-up class, as you asked"
                    + also + ". Your sections are told once the timetable office approves the change.")
        closes = any(c.type == ConstraintType.UNAVAILABLE and c.scope.room for c in case.constraints)
        if cancelled and closed and closes:  # this request is the closure (others in its week are not)
            return (f"Recorded: {' and '.join(closed)} closed in week {week}. "
                    f"{len(cancelled)} {'class has' if len(cancelled) == 1 else 'classes have'} no other room "
                    "that week and will be cancelled, with a make-up owed"
                    + also + ". Their teachers and sections are told once the timetable office approves the change.")
        if not moved and not cancelled and (under := self.builds_on(case)):
            return (f"Your request is met by {under.id}, which is waiting for the coordinator's approval: "
                    "yours is approved after it, and nothing else has to move.")
        if agreed := self._agreed_text(case, outcome, assignment, moved):
            return f"{agreed} {waiting}"
        if chosen:
            return f"{chosen} {waiting}"
        if notices := outcome.notices.get(case.request.sender_id):
            return " ".join(notices)
        return f"Your request can be met. {waiting}"

    def _agreed_text(self, case: Case, outcome, assignment: Mapping[str, Placement], moved: list[str]) -> str | None:
        """What a sender who gave something up in the negotiation ends up with: where their classes are, and what
        they first asked for instead. None when they gave nothing up ("can be met" says it)."""
        sender = case.request.sender_id
        given = {e.constraint_id for e in outcome.concessions if e.stakeholder == sender}
        if outcome.status != "agreed" or not given:
            return None
        cons = {c.id: c for c in self.store.constraints(active_only=False)}
        asked = [_when(self.instance, cons[cid]) for cid in sorted(given) if cid in cons and cons[cid].when.days]
        sessions = self.instance.session_by_id
        mine = [s for s in moved if s in assignment and sessions[s].faculty == sender]
        where = "; ".join(f"{session_name(self.instance, s)} is now on {placement_text(self.instance, assignment[s], s)}"
                          for s in mine)
        if not where:
            return None
        return (f"Agreed after negotiation: {where[0].upper() + where[1:]}"
                + (f", instead of {' and '.join(asked)} as you first asked." if asked else "."))

    def _let_sender_choose(self, case: Case, assignment: Mapping[str, Placement], baseline, week: int | None,
                           neg: Negotiator, cancelled: Iterable[str]) -> tuple[Mapping[str, Placement], str | None]:
        """A request that fits in more than one way: the sender picks the time. Up to ``CHOICES``
        solver-checked timetables that place the sender's moved classes at different times (not
        only in other rooms) and move no more classes than the solver's own answer; one option,
        or no one to ask, keeps that answer. No choice by the deadline takes option A."""
        sender = case.request.sender_id
        sessions = self.instance.session_by_id
        moved = moved_sessions(baseline or {}, assignment)
        own = [sid for sid in moved if sid in sessions and sessions[sid].faculty == sender]
        if not own or sender not in self.responders or self.offer_choices is None or not self.offer_choices():
            return assignment, None
        options = [dict(assignment)]
        cons, forbid = self.store.constraints(), []
        while len(options) < CHOICES:
            for sid in own:  # this time is taken: the next option puts every one of them elsewhere
                p = options[-1][sid]
                forbid.append(Constraint(id=f"CHOICE-X-{sid}-{p.day}{p.slot}", type=ConstraintType.AVOID, hard=True,
                                         tier=Tier.OPERATIONAL, scope=Scope(session=sid),
                                         when=When(days=[p.day], slots=[p.slot])))
            res = TimetableSolver(self.instance, [*cons, *forbid], week=week, baseline=baseline,
                                  time_limit=neg.time_limit, presolve=neg.presolve, workers=neg.workers,
                                  seed=neg.seed, skip=frozenset(cancelled)).solve()
            if not res.ok or len(moved_sessions(baseline or {}, res.assignment)) > len(moved):
                break
            options.append(dict(res.assignment))
        if len(options) < 2:
            return assignment, None

        def when(a: Mapping[str, Placement]) -> str:
            return "; ".join(f"{session_name(self.instance, sid)} on {placement_text(self.instance, a[sid])}"
                             for sid in own)

        keys = list("ABCDEFGH"[:len(options)])
        offers = [Offer(key=k, drop=[], placements={sid: a[sid] for sid in own},
                        moved=len(moved_sessions(baseline or {}, a))) for k, a in zip(keys, options)]
        others = len(moved) - len(own)
        text = ("Your request can be met in more than one way. Which time do you prefer? Each option has been "
                "checked by the solver and " + (f"moves {others} other class{'' if others == 1 else 'es'}."
                                                if others else "moves no other class.")
                + " Without an answer by the deadline, option A is used.\n\nOptions:\n"
                + "\n".join(f"{k}) {when(a)}" for k, a in zip(keys, options)))
        message = Message(round=0, to=sender, mus=[], offers=offers, text=text, explanation_mode="choice",
                          faithfulness=1.0)
        self.store.log(case.id, "choice_offered", to=sender, options={k: when(a) for k, a in zip(keys, options)})
        reply = self.responders[sender].respond(message)
        picked = (reply.choice or "").strip().upper() if reply.decision == "accept" else ""
        k = picked if picked in keys else "A"
        self.store.log(case.id, "choice_made", choice=k, reply=reply.decision, defaulted=k != picked)
        a = options[keys.index(k)]
        said = (f"You chose option {k}: {when(a)}." if k == picked else
                f"No time was chosen, so option A is used: {when(a)}.")
        return a, said

    def _audit(self, *outcomes: Outcome) -> dict:
        teaching = sorted({s.faculty for s in self.instance.sessions})
        ledger = ConcessionLedger(self.store.ledger())
        made = [e for o in outcomes for e in o.concessions]
        before = ConcessionLedger([e for e in ledger.entries if e not in made])
        return {"gini_before": round(before.gini(teaching, self.semester), 4),
                "gini_after": round(ledger.gini(teaching, self.semester), 4),
                "conceded": sorted({e.stakeholder for e in made}),
                "preferences_lost": sorted({p for o in outcomes for p in o.notices})}

    def _withdraw(self, case: Case) -> None:
        for c in case.constraints:
            self.store.set_active(c.id, False)
        for cid in case.superseded:
            self.store.set_active(cid, True)

    # -- escalations: a person with the authority decides ------------------------------------

    def overrides(self, case: Case) -> list[Constraint]:
        """What granting an escalated request sets aside: the other constraints in its conflicts,
        except physical facts (a closed room stays closed)."""
        if case.outcome is None:
            return []
        own = {c.id for c in case.constraints}
        known = {c.id: c for c in self.store.constraints(active_only=False)}
        ids = dict.fromkeys(i for mus in case.outcome.mus_log for i in mus if i not in own)
        return [known[i] for i in ids if i in known and known[i].tier != Tier.PHYSICAL]

    def cannot_grant(self, case: Case) -> str | None:
        """Why granting this escalated request would not help, or None when it can be granted."""
        if case.extra is not None or case.routing.get("outside_hours"):
            return None  # placed when granted; or held outside the timetable, arranged by the office
        if case.parse is None or not case.parse.constraints:
            return "this request has nothing to put into the timetable"
        exception = case.outcome is not None and case.outcome.escalation is not None and case.outcome.escalation.rule
        if case.outcome is not None and not exception and not self.overrides(case):  # a deadlock nothing would undo
            return ("it cannot fit even on its own, so granting would not help; "
                    "decline it, or ask the sender for a narrower request")
        return None

    @_one_at_a_time
    def decide(self, case_id: str, decider: str, grant: bool, note: str = "", name: str | None = None) -> Case:
        """Grant (the request goes back through the solver with its conflicts set aside, and then to
        approval like any change) or decline (the sender is told, with the note)."""
        case = self.cases[case_id]
        if case.status != S.ESCALATED:
            raise ValueError(f"case {case_id} is not escalated ({case.status.value})")
        name = name or (self.instance.faculty_by_id[decider].name if decider in self.instance.faculty_by_id else decider)
        if not grant:
            case.decision = {"by": decider, "granted": False, "note": note}
            case.reply = f"Your request was declined by {name}" + (f": {note}" if note else ".")
            self._advance(case, S.DENIED, decided_by=decider, note=note)
            return case
        if case.routing.get("outside_hours"):  # held outside the timetable: the office arranges it
            case.decision = {"by": decider, "granted": True, "note": note, "set_aside": []}
            self._advance(case, S.COMPILED, decided_by=decider, granted=True, note=note, set_aside=[])
            case.reply = (f"Approved by {name}: the class may be held outside regular hours as you asked"
                          + (f" ({note})" if note else "") + ". It is outside the weekly timetable, so the timetable "
                          "office books the room and tells your students.")
            self._advance(case, S.ANSWERED, outside_hours=True, decided_by=decider)
            return case
        if why := self.cannot_grant(case):
            raise ValueError(why)
        if case.extra is not None:
            case.decision = {"by": decider, "granted": True, "note": note, "set_aside": []}
            self._advance(case, S.COMPILED, decided_by=decider, granted=True, note=note, set_aside=[])
            return self._place_extra(case)
        set_aside = self.overrides(case)
        if not case.constraints:  # sent up by the policy check before it was compiled
            case.constraints = (case.parse.constraints if case.compiled is not None else
                            self._on_named_day(case, case.parse.constraints))
            case.superseded = self._supersede(case)
            self.store.add_constraints(case.constraints, request_id=case.id)
        for c in case.constraints:
            self.store.set_active(c.id, True)
        for cid in case.superseded:
            self.store.set_active(cid, False)
        for c in set_aside:
            self.store.set_active(c.id, False)
        case.decision = {"by": decider, "granted": True, "note": note, "set_aside": [c.id for c in set_aside]}
        self._advance(case, S.COMPILED, decided_by=decider, granted=True, note=note,
                      set_aside=[c.id for c in set_aside])
        self._solve(case)
        if case.status == S.ESCALATED:  # still no way: put back what was set aside
            for c in set_aside:
                self.store.set_active(c.id, True)
            case.decision = None
        return case

    def _orders_colleague_class(self, case: Case) -> str | None:
        """The colleague whose class a teacher tells the system to change ("Move Dr. Iyer's Friday lecture",
        "Cancel Dr. Rao's lab"), whatever the reader made of it: a reading that drops the name would change
        the sender's own class instead, and one that keeps it changes a class the sender has no authority
        over. A swap is asked of the colleague, and the head of department has the department's classes, so
        neither is stopped here."""
        if case.request.role not in (Role.FACULTY, Role.GUEST_FACULTY) or case.intents(SwapIntent):
            return None
        for m in ORDERS_COLLEAGUE.finditer(case.request.raw_text):
            name = m.group(1).lower()
            for f in self.instance.faculty:
                if f.id != case.request.sender_id and surname(f.name).lower() == name:
                    return f.id
        return None

    def _refuse_colleague(self, case: Case, colleague: str) -> Case:
        why = "faculty can only change their own classes"
        case.parse.refusal = why
        case.route += "+authority"
        case.reply = (f"Your request was not processed: {why}. To change {self.instance.faculty_by_id[colleague].name}'s "
                      "class, ask them for a swap, or ask the timetable office. The timetable coordinator has been informed.")
        self._advance(case, S.REFUSED, reason=why, colleague=colleague)
        return case

    def _class_rep_report(self, case: Case) -> bool:
        """A class representative's message about the timetable is a report for the timetable office, not a
        request: the office looks into it and, if a change is needed, makes it as its own request (which asks
        the teacher concerned, and goes to the head of department when they disagree). So it is passed on
        whole, never compiled or refused for want of authority. Questions are still answered, club bookings
        still go to the club desk, a clash the reader found is still forwarded with the clash (``_clash``),
        and a refusal for any other reason (an injection) stands."""
        if case.request.role != Role.STUDENT or case.intents(ClubIntent):
            return False
        action = case.parse.action
        if action in (ExpectedAction.ANSWER, ExpectedAction.OUT_OF_SCOPE):
            return False
        if action == ExpectedAction.INVESTIGATE and case.compiled is not None and case.compiled.clashes:
            return False
        return case.parse.refusal in (None, STUDENTS_REPORT)

    def _report(self, case: Case) -> Case:
        case.route += "+report"
        case.reply = ("Thank you. Your message has been passed to the timetable office, which will look into it "
                      "and, if a change is needed, work it out with the teacher or the head of department. "
                      "You will hear back here.")
        self._advance(case, S.FORWARDED, to="coordinator", report="class_rep")
        return case

    def office_answers(self, case: Case) -> bool:
        """A message the timetable office answers or acts on: one forwarded to it, or one refused (a student
        asking for a change, a colleague's class): the refusal stands, but the office may still reply, or
        make the change itself as its own request."""
        return case.forwarded_to == "coordinator" or case.status == S.REFUSED

    def act_on(self, case_id: str, by: str, request: Request) -> Case:
        """The coordinator takes up a message forwarded to them (a student's clash report, a change only the
        office can make) as their own ``request``, which the caller then submits: it goes through the whole
        pipeline with the coordinator's authority. The forwarded message is answered with a pointer to it."""
        case = self.cases[case_id]
        if by not in self.approvers:
            raise NotAuthorised(f"{by} cannot act for the timetable office")
        if not self.office_answers(case):
            raise ValueError(f"case {case_id} was neither forwarded to the coordinator nor refused ({case.status.value})")
        if case.handled is not None:
            raise ValueError(f"case {case_id} has already been answered")
        note = f"The timetable office is acting on your message as request {request.id}."
        case.handled = {"by": by, "note": note, "at": datetime.now().isoformat(timespec="seconds"),
                        "acting": request.id}
        case.reply = note
        self.store.log(case.id, "acted_on", by=by, request=request.id)
        return case

    def handle(self, case_id: str, handler: str, note: str, name: str | None = None) -> Case:
        """The coordinator answers a message forwarded to them; the sender gets the answer as the reply.
        The request stays forwarded: nothing in it reaches the solver."""
        case = self.cases[case_id]
        if not self.office_answers(case):
            raise ValueError(f"case {case_id} was neither forwarded to the coordinator nor refused ({case.status.value})")
        if case.handled is not None:
            raise ValueError(f"case {case_id} has already been answered")
        note = note.strip()
        if not note:
            raise ValueError("write a reply first")
        name = name or (self.instance.faculty_by_id[handler].name if handler in self.instance.faculty_by_id else handler)
        case.handled = {"by": handler, "note": note, "at": datetime.now().isoformat(timespec="seconds")}
        case.reply = f"Reply from {name}: {note}"
        self.store.log(case.id, "handled", by=handler, note=note)
        return case

    # -- withdrawal by the sender --------------------------------------------------------

    def withdrawable(self, case: Case) -> str | None:
        """Why ``case`` cannot be withdrawn, or None when it can: only while nothing it asked for is in
        the published timetable. A published change is undone by a new request, not a withdrawal."""
        s = case.status
        if s in (S.CLARIFICATION, S.ESCALATED, S.AWAITING_APPROVAL):
            return None
        if s == S.FORWARDED:
            return "the timetable office has already answered it" if case.handled else None
        if s == S.PUBLISHED:
            return "it has changed the published timetable" if case.proposal is not None else None
        if s in (S.REFUSED, S.DENIED, S.ANSWERED, S.WITHDRAWN):
            return f"it is already closed ({s.value})"
        return "it is still being worked on"

    def withdraw(self, case_id: str, by: str) -> Case:
        """Take a request back and undo everything it set in motion: its constraints are switched off and
        the ones it replaced switched back on; for one that was solved (awaiting approval, or recorded
        because it already fitted), the preferences its negotiation set aside come back and the
        concessions it recorded leave the ledger. Its proposed version is never published."""
        case = self.cases[case_id]
        if why := self.withdrawable(case):
            raise ValueError(f"case {case_id} cannot be withdrawn: {why}")
        own, restored, refunded = self._undo(case)
        # people who gave something up for it hear that it no longer stands
        case.notices = {p: f"{case.id} was withdrawn by its sender; what you agreed to for it no longer applies."
                        for p in sorted(set(refunded))}
        case.reply = "You withdrew this request. The timetable is as it was before it."
        self._advance(case, S.WITHDRAWN, by=by, constraints=sorted(own), restored=restored,
                      refunded=sorted(set(refunded)), version=case.proposal.version if case.proposal else None)
        return case

    def _undo(self, case: Case) -> tuple[set[str], list[str], list[str]]:
        """Switch off what ``case`` set in motion (see ``withdraw``): its constraints, ids of the ones
        restored and the people whose concessions were refunded."""
        solved = case.status in (S.AWAITING_APPROVAL, S.PUBLISHED)
        own = set(self.store.request_constraints(case.id)) | {c.id for c in case.constraints}
        for cid in own:
            self.store.set_active(cid, False)
        self.store.set_extras_active(case.id, False)
        for cid in case.superseded:
            self.store.set_active(cid, True)
        restored, refunded = [], []
        if solved and case.outcome is not None:
            for cid in case.outcome.relaxed:
                if cid not in own:
                    self.store.set_active(cid, True)
                    restored.append(cid)
            for e in case.outcome.concessions:
                if self.store.remove_concession(e):
                    refunded.append(e.stakeholder)
        return own, restored, refunded

    # -- human approval (L6)-------------------------------------------------------------

    def pending(self) -> list[Case]:
        return [c for c in self.cases.values() if c.status == S.AWAITING_APPROVAL]

    def _tip(self, week: int | None, follow_semester: bool = True) -> TimetableVersion | None:
        """What a new change builds on: the newest change for ``week`` still waiting for approval (each one
        builds on the one before), else the published timetable. A change then shows only its own moves,
        and is published after the ones it builds on. A week with no repair of its own yet is the semester
        timetable, so its first repair builds on the semester's newest change still waiting: the constraints
        of that change are in force, and built on the published timetable the repair would make its moves
        again as moves of its own (``follow_semester``; not for an extra class, which goes into a free hour
        without the solver and moves nothing)."""
        waiting = [c.proposal for c in self.pending() if c.proposal is not None and c.proposal.week == week]
        if not waiting and follow_semester and week is not None and self.store.current_version(week) is None:
            waiting = [c.proposal for c in self.pending() if c.proposal is not None and c.proposal.week is None]
        if waiting:
            return max(waiting, key=lambda v: v.version)
        return self.store.current_version(week) or self.store.current_version()

    def _cancelled(self, week: int | None) -> tuple[set[str], list[str]]:
        """Classes not held in ``week`` (a room closure leaves them nowhere to go, or their teacher is away
        and asked for no make-up), and the closed rooms."""
        closed, rooms = closures(self.instance, self.store.constraints(), week)
        return closed | self._absent(week), rooms

    def _absent(self, week: int | None, only: str | None = None) -> set[str]:
        """Classes cancelled in ``week`` because their teacher is away then (an absence in given weeks) and
        did not ask for make-up classes; ``only``: for that request's absence alone. Where a class would be
        that week is the semester timetable's placement."""
        semester = self.store.current_version()
        if week is None or semester is None:
            return set()
        placed = _regular(semester.assignment)
        out = set()
        for c in self.store.constraints():
            case = self.cases.get(c.source.request or "")
            if (case is None or case.request.makeup is not False or (only and case.id != only)
                    or c.type != ConstraintType.UNAVAILABLE or not c.hard or not (c.scope.faculty or c.scope.session)
                    or not c.when.weeks or week not in c.when.weeks):
                continue
            if c.scope.session:  # one class cancelled ("cancel my practical in week 12")
                if c.scope.session in placed:
                    out.add(c.scope.session)
                continue
            for s in self.instance.sessions:
                p = placed.get(s.id)
                if s.faculty == c.scope.faculty and p is not None and violates(
                        self.instance, c, s, self.instance.day_index(p.day), p.slot, self.instance.room_by_id[p.room]):
                    out.add(s.id)
        return out

    def builds_on(self, case: Case) -> Case | None:
        """The change still waiting for approval that ``case``'s proposal was built on, if any."""
        parent = case.proposal.parent if case.proposal else None
        return next((c for c in self.pending() if c.proposal is not None and c.proposal.version == parent), None)

    def approve(self, case_id: str, approver: str) -> TimetableVersion | None:
        """Publish the case's proposal. A requirement the timetable already met (``_already_fits``) has no
        proposal: it is confirmed and stays in force, and None is returned."""
        if approver not in self.approvers:
            self.store.log(case_id, "approval_refused", approver=approver)
            raise NotAuthorised(f"{approver} may not approve timetable changes")
        case = self.cases[case_id]
        if case.status != S.AWAITING_APPROVAL:
            raise ValueError(f"case {case_id} is not awaiting approval ({case.status.value})")
        if case.proposal is None:
            case.reply = ("The timetable office confirmed it. Nothing needed to move; later changes will "
                          "keep to it.")
            self._advance(case, S.PUBLISHED, version=None, moved=[], approver=approver, record_only=True)
            return None
        if under := self.builds_on(case):
            raise ValueError(f"this change builds on {under.id}, which is still waiting for approval: "
                             f"approve or reject {under.id} first")
        # built on a change that was published since, or on one that was rejected or withdrawn: solved again
        before = self.store.current_version(case.proposal.week) or self.store.current_version()
        if before is not None and case.proposal.parent != before.version:
            case.proposal = self._rebase(case, before)
        v = self.store.publish_version(case.proposal.version, approver)
        moved = moved_sessions(before.assignment if before else {}, v.assignment)
        g = build_graph(self.instance)
        regular = [s for s in moved if not s.startswith(EXTRA_PREFIX)]
        add_change(g, f"V{v.version}", regular)
        case.notices = self._notices(g, f"V{v.version}", regular, v)
        for person, text in self._extra_notices(v, moved).items():
            case.notices[person] = f"{case.notices[person]} {text}" if person in case.notices else text
        cancelled = [(v.week, self._cancellations(v, before), before)]  # worded together after the last week
        for i, w in enumerate(case.more):  # the request's further weeks, each on its week's live timetable
            live = self.store.current_version(w.week) or self.store.current_version()
            if live is not None and w.parent != live.version:
                w = case.more[i] = self._rebase(case, live, w)
            published = self.store.publish_version(w.version, approver)
            moved_w = [s for s in moved_sessions(live.assignment if live else {}, published.assignment)
                       if not s.startswith(EXTRA_PREFIX)]
            gw = build_graph(self.instance)
            add_change(gw, f"V{published.version}", moved_w)
            for person, text in self._notices(gw, f"V{published.version}", moved_w, published).items():
                case.notices[person] = f"{case.notices[person]} {text}" if person in case.notices else text
            cancelled.append((published.week, self._cancellations(published, live), live))
        for person, text in self._cancellation_text(cancelled).items():
            case.notices[person] = f"{case.notices[person]} {text}" if person in case.notices else text
        self._advance(case, S.PUBLISHED, version=v.version, approver=approver, notified=sorted(case.notices),
                      **({"more": [w.version for w in case.more]} if case.more else {}))
        if v.week is None:
            self.carry_into_weeks(v, approver)
        if self.on_published is not None:
            try:
                self.on_published(case)
            except Exception as e:  # noqa: BLE001 - telling people must never undo a publish
                self.store.log(case.id, "notify_failed", error=f"{type(e).__name__}: {e}")
        return v

    @_one_at_a_time
    def carry_into_weeks(self, v: TimetableVersion, approver: str) -> list[TimetableVersion]:
        """A week with its own repair (an absence in week 7) shows that repair, not the semester
        timetable; a semester change published later must reach it too. Each such week is solved
        again on top of ``v`` with that week's constraints, and published with ``v``'s approval."""
        out = []
        weeks = sorted({x["week"] for x in self.store.versions() if x["published"] and x["week"] is not None})
        for week in weeks:
            cancelled, _ = self._cancelled(week)
            result = TimetableSolver(self.instance, self.store.constraints(), week=week, baseline=v.assignment,
                                     time_limit=30, skip=cancelled).solve()
            if not result.ok:
                self.store.log(None, "carry_failed", version=v.version, week=week, status=result.status)
                continue
            w = self.store.propose_version(result.assignment, case_id=f"carry-{v.version}", week=week,
                                           cancelled=sorted(cancelled))
            out.append(self.store.publish_version(w.version, approver))
            self.store.log(None, "carried", version=v.version, week=week, into=w.version)
        return out

    @_one_at_a_time
    def _rebase(self, case: Case, base: TimetableVersion, stale: TimetableVersion | None = None) -> TimetableVersion:
        """Another change was published after this proposal (``stale``, default the case's) was made.
        Publishing it as it is would put back the timetable it was built on and undo that change, so solve
        it again from the live version, with every constraint in force (this request's included)."""
        stale = stale or case.proposal
        week = stale.week
        cancelled, _ = self._cancelled(week)
        result = TimetableSolver(self.instance, self.store.constraints(), week=week,
                                 baseline=_regular(base.assignment), time_limit=30, skip=cancelled).solve()
        if not result.ok:
            raise ValueError("the timetable has changed since this proposal and it no longer fits; "
                             "reject it and ask the requester to send it again")
        v = self.store.propose_version(result.assignment, case.id, week=week, cancelled=sorted(cancelled))
        self.store.log(case.id, "rebased", stale=stale.version, on=base.version, version=v.version)
        return v

    def _already_met(self, case: Case, base: TimetableVersion, week: int | None) -> bool:
        """The live timetable keeps every hard rule in force and this request's own wishes too."""
        if verify_timetable(self.instance, _regular(base.assignment), self.store.constraints(), week):
            return False
        for c in case.constraints:
            if c.type not in PLACEMENT_TYPES:
                return False  # not checked directly: let the solver decide
            for s in sessions_in_scope(self.instance, c.scope):
                p = base.assignment.get(s.id)
                if p is not None and violates(self.instance, c, s, self.instance.day_index(p.day), p.slot,
                                              self.instance.room_by_id[p.room]):
                    return False
        return True

    def _extra_notices(self, v: TimetableVersion, moved: list[str]) -> dict[str, str]:
        """The teacher and the sections of an extra class that ``v`` adds or moves."""
        out: dict[str, str] = {}
        for x in self.store.extras(v.week) if v.week is not None else []:
            if x.session.id in moved and x.session.id in v.assignment:
                text = (f"Extra class: {self.extra_reader.label(x.session)} on "
                        f"{when_text(self.instance, v.assignment[x.session.id], x.week)}.")
                for person in [x.session.faculty, *(f"ST-{g}" for g in x.session.groups)]:
                    out[person] = f"{out[person]} {text}" if person in out else text
        return out

    def cancel_reasons(self, week: int | None) -> tuple[dict[str, str], list[str]]:
        """Why each class not held in ``week`` is not held: "away" (its teacher is away and asked for no make-up),
        "asked" (its teacher asked for that class to be cancelled), "closed" (its room is closed and no other room
        has what it needs); and the closed rooms."""
        closed, rooms = closures(self.instance, self.store.constraints(), week)
        asked: set[str] = set()
        for c in self.store.constraints():
            case = self.cases.get(c.source.request or "")
            if (case is not None and case.request.makeup is False and c.type == ConstraintType.UNAVAILABLE and c.hard
                    and c.scope.session and not c.scope.faculty and c.when.weeks and week in c.when.weeks):
                asked.add(c.scope.session)
        away = self._absent(week) - asked
        return {**{s: "closed" for s in closed}, **{s: "away" for s in away}, **{s: "asked" for s in asked}}, rooms

    def _cancellations(self, v: TimetableVersion, before: TimetableVersion | None) -> dict[str, dict[str, tuple]]:
        """Who ``v`` newly cancels a class for in its week: person -> reason ("away", "asked", or "closed:" with the
        rooms) -> the classes. A class cancelled already in ``before`` (the week's timetable ``v`` replaces) was told
        then."""
        told = set(before.cancelled or []) if before is not None and before.week == v.week else set()
        cancelled = [sid for sid in v.cancelled or [] if sid not in told]
        if not cancelled:
            return {}
        reasons, rooms = self.cancel_reasons(v.week)
        out: dict[str, dict[str, list[str]]] = {}
        for sid in cancelled:
            s = self.instance.session_by_id[sid]
            why = reasons.get(sid, "away")
            reason = "closed:" + " and ".join(rooms) if why == "closed" else why
            for person in [s.faculty, *(f"ST-{g}" for g in s.groups)]:
                out.setdefault(person, {}).setdefault(reason, []).append(sid)
        return {person: {r: tuple(sorted(sids)) for r, sids in by.items()} for person, by in out.items()}

    def _cancellation_text(self, weeks: list[tuple[int | None, dict[str, dict[str, tuple]], TimetableVersion | None]]
                           ) -> dict[str, str]:
        """The cancellations of one or more weeks in a few words a person. Weeks that cancel the same classes for the
        same reason are one sentence, and classes of one course one phrase with their times ("Weeks 10-12: your
        Computer Networks lectures (Monday 10 am and 12 pm, Friday 3 pm) are cancelled while you are away. No make-up
        classes."); a section reads the teacher's name once ("... are cancelled: Dr. Das is away.")."""
        groups: dict[str, dict[tuple[str, tuple], list[int]]] = {}
        where: dict[str, object] = {}  # a class's usual place, from the week's timetable it is cancelled from
        for week, by_person, before in weeks:
            for person, by in by_person.items():
                for reason, sids in by.items():
                    groups.setdefault(person, {}).setdefault((reason, sids), []).append(week)
                    for sid in sids:
                        if before is not None and sid in before.assignment:
                            where.setdefault(sid, before.assignment[sid])
        out: dict[str, str] = {}
        for person, by in groups.items():
            parts = []
            for (reason, sids), wks in by.items():
                teachers = {self.instance.session_by_id[sid].faculty for sid in sids}
                mine = person in teachers
                courses = self._courses_named(sids, where)
                n = len(sids)
                teacher = self.instance.faculty_by_id[next(iter(teachers))].name if len(teachers) == 1 and \
                    next(iter(teachers)) in self.instance.faculty_by_id else None
                what = f"{'your ' if mine else ''}{courses}"
                what = what[0].upper() + what[1:] if not mine else what
                verb = "is" if n == 1 else "are"
                none = f"No make-up {'class' if n == 1 else 'classes'}."
                if reason == "away":
                    tail = (f"while you are away. {none}" if mine else
                            f"{teacher} is unavailable. {none}" if teacher else f"their teachers are unavailable. {none}")
                    joint = " " if mine else ": "
                elif reason == "asked":
                    tail = f"as you asked. {none}" if mine else f"{teacher or 'their teacher'} asked for it. {none}"
                    joint = ", " if mine else ": "
                else:
                    rooms = reason.removeprefix("closed:") or "the room"
                    tail = (f"because {rooms} {'is' if ' and ' not in rooms else 'are'} closed and no other room has "
                            f"what {'it needs' if n == 1 else 'they need'}. A make-up class is owed.")
                    joint = " "
                parts.append(f"{_weeks_named(wks)}: {what} {verb} cancelled{joint}{tail}")
            out[person] = " ".join(parts)
        return out

    def _courses_named(self, sids: tuple, where: dict) -> str:
        """Classes by course, each with its times, in time order: "Computer Networks lectures (Monday 10 am and
        12 pm, Friday 3 pm) and Digital Logic lecture (Friday 9 am)"."""
        def at(sid: str):
            p = where.get(sid)
            return (DAYS.index(p.day), p.slot) if p is not None else (len(DAYS), 0)

        by_course: dict[tuple[str, str], list[str]] = {}
        for sid in sorted(sids, key=at):
            s = self.instance.session_by_id[sid]
            by_course.setdefault((self.instance.course_title(s.course), s.kind.value), []).append(sid)
        phrases = []
        for (title, kind), ids in by_course.items():
            days: dict[str, list[str]] = {}
            for sid in ids:
                p = where.get(sid)
                if p is not None:
                    days.setdefault(FULL_DAY.get(p.day, p.day), []).append(hour(p.slot))
            times = ", ".join(f"{d} {_listed(hs)}" for d, hs in days.items())
            plural = {"lecture": "lectures", "practical": "practicals", "tutorial": "tutorials"}.get(kind, f"{kind}s")
            phrases.append(f"{title} {plural if len(ids) > 1 else kind}{f' ({times})' if times else ''}")
        return _listed(phrases)

    def reject(self, case_id: str, approver: str, reason: str = "") -> None:
        if approver not in self.approvers:
            raise NotAuthorised(f"{approver} may not review timetable changes")
        case = self.cases[case_id]
        if case.status == S.AWAITING_APPROVAL and case.proposal is None:
            # a requirement that moved nothing: there is nothing to negotiate again, so it is declined
            own, restored, _ = self._undo(case)
            why = f": {reason.strip().rstrip('.')}." if reason.strip() else "."
            case.reply = (f"The timetable office did not confirm this{why} It is not in force; send it again in "
                          "other words if it was misread.")
            self._advance(case, S.DENIED, rejected_by=approver, reason=reason, constraints=sorted(own),
                          restored=restored)
            return
        self._advance(case, S.NEGOTIATING, rejected_by=approver, reason=reason)

    def _notices(self, g, change: str, moved: list[str], v: TimetableVersion) -> dict[str, str]:
        people = affected_stakeholders(g, change)
        out = {}
        for person in sorted(people):
            mine = [s for s in moved if s in v.assignment and (
                any(x.id == s and x.faculty == person for x in self.instance.sessions) or
                any(x.id == s and f"ST-{grp}" == person for x in self.instance.sessions for grp in x.groups))]
            if mine:
                week = f" (week {v.week})" if v.week else ""
                out[person] = "Timetable change" + week + ": " + "; ".join(
                    f"{session_name(self.instance, s)} is now on {placement_text(self.instance, v.assignment[s], s)}"
                    for s in mine)
        return out


def _regular(assignment: Mapping[str, Placement]) -> dict[str, Placement]:
    """An assignment without its extra classes (the solver and the validator know only weekly sessions)."""
    return {s: p for s, p in assignment.items() if not s.startswith(EXTRA_PREFIX)}


def _weeks_text(weeks: list[int | None]) -> str:
    """"week 7", "weeks 5-8", "weeks 3, 5 and 9"."""
    weeks = sorted({w for w in weeks if w is not None})
    if len(weeks) == 1:
        return f"week {weeks[0]}"
    if weeks == list(range(weeks[0], weeks[-1] + 1)):
        return f"weeks {weeks[0]}-{weeks[-1]}"
    return "weeks " + ", ".join(map(str, weeks[:-1])) + f" and {weeks[-1]}"


def _spread(found: list[Placement], n: int = 4) -> list[str]:
    """Up to ``n`` free times, one per day first."""
    picked, days = [], set()
    for p in found:
        if p.day not in days:
            picked.append(p)
            days.add(p.day)
    picked += [p for p in found if p not in picked]
    return [f"{FULL_DAY[p.day]} at {hour(p.slot)}" for p in picked[:n]]


def _plan_view(plan: SwapPlan) -> dict:
    return {"requester": plan.requester, "counterpart": plan.counterpart, "mine": plan.mine, "theirs": plan.theirs,
            "mine_at": plan.mine_at.model_dump(), "theirs_at": plan.theirs_at.model_dump(), "weeks": plan.weeks}


def explain_constraint(instance: Instance, c: Constraint) -> str:
    """What the coordinator sees for a constraint (no private reasons)."""
    return describe(instance, c)


_DAY_WORD = re.compile(r"\b(mon|tue|tues|wed|thu|thur|thurs|fri|sat)(?:day|nesday|sday|urday|rsday)?s?\b", re.IGNORECASE)
_RELEASE = re.compile(r"\b(?:is|are)\s+(?:now\s+)?(?:fine|ok|okay|alright|free to use)\b|\bno longer\b|\bnot needed\b|"
                      r"\bdon'?t need\b|\bcan be used\b|\bnot\s+(?:on\s+)?(?=(?:mon|tue|wed|thu|fri))", re.IGNORECASE)
_AMENDS = re.compile(r"\b(?:actually|earlier|previous|any ?more|now|no longer|change my|update my)\b", re.IGNORECASE)
_KEEP = re.compile(r"\b(?:only|just|still)\b", re.IGNORECASE)
_DAY_OF = {"mon": "Mon", "tue": "Tue", "wed": "Wed", "thu": "Thu", "fri": "Fri", "sat": "Sat"}


_PLACES = re.compile(r"\b(?:move|moved|shift|shifted|reschedule|rescheduled|schedule|scheduled|hold|held|put|have|"
                     r"take|teach|extra|make-?up|be on|be at)\b", re.IGNORECASE)
_NOT = re.compile(r"\b(?:no|not|never|can'?t|cannot|avoid|free|unavailable|away|without|don'?t)\b", re.IGNORECASE)
_AT_CLOCK = re.compile(r"\b(?:at|to|from|for|by)\s+(\d{1,2})(?::(\d\d))?\s*(am|pm)\b", re.IGNORECASE)
_WEEKEND = re.compile(r"\b(?:to|on|for)\s+(?:a\s+|the\s+)?(saturday|sunday)s?\b", re.IGNORECASE)


def _asks_outside_hours(text: str, instance: Instance) -> str | None:
    """"... moved to Thursday at 6 pm", "... held on Saturday morning": a class placed at a time the timetable
    does not have. Only in a sentence that places a class and does not rule a time out ("no classes on
    Saturday" is a wish about the weekend, not a request to teach then)."""
    cal = instance.calendar
    for sentence in re.split(r"(?<=[.!?;])\s+", text):
        if not _PLACES.search(sentence) or _NOT.search(sentence):
            continue
        for m in _AT_CLOCK.finditer(sentence):
            h = int(m[1]) % 12 + (12 if m[3].lower() == "pm" else 0)
            if h not in {int(hour(q).split()[0]) % 12 + (12 if hour(q).endswith("pm") else 0)
                         for q in range(cal.slots_per_day)}:
                return m[0].split(None, 1)[1]
        if (m := _WEEKEND.search(sentence)) and m[1][:3].title() not in cal.days:
            return m[1].title()
    return None


def _weeks_named(weeks: list[int | None]) -> str:
    """[10, 11, 12] -> "Weeks 10-12"; [10, 12] -> "Weeks 10 and 12"; [10] -> "Week 10"; [None] -> "Every week"."""
    ws = sorted({w for w in weeks if w is not None})
    if not ws:
        return "Every week"
    if len(ws) == 1:
        return f"Week {ws[0]}"
    if ws == list(range(ws[0], ws[-1] + 1)):
        return f"Weeks {ws[0]}–{ws[-1]}"
    return f"Weeks {_listed([str(w) for w in ws])}"


def _listed(items: list[str]) -> str:
    """"a", "a and b", "a, b and c"."""
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def _request_number(request_id: str | None) -> int:
    """R-00015 -> 15 (R-00003-2, a part of a request, -> 3): which request came first. 0 when there is none."""
    m = re.match(r"R-(\d+)", request_id or "")
    return int(m[1]) if m else 0


def _amended_days(text: str) -> tuple[set[str], set[str]]:
    """The days a message lets go of and the days it keeps, clause by clause: "Wednesday mornings are fine
    now" lets go of Wednesday; "only Monday mornings need to stay free" keeps Monday; "not Wednesday" lets go."""
    released: set[str] = set()
    kept: set[str] = set()
    for clause in re.split(r"[;.,!?]|\bbut\b|\band\s+(?=only\b)", text, flags=re.IGNORECASE):
        for m in re.finditer(r"\bnot\s+(?:on\s+)?" + _DAY_WORD.pattern, clause, flags=re.IGNORECASE):
            released.add(_DAY_OF[m[1][:3].lower()])
        days = {_DAY_OF[m[1][:3].lower()] for m in _DAY_WORD.finditer(clause)} - released
        if _RELEASE.search(clause):
            released |= days
        elif _KEEP.search(clause):
            kept |= days
    return released, kept - released


def _short(text: str, n: int = 60) -> str:
    """A message's first words, for naming a request in a reply."""
    one = " ".join(text.split())
    return one if len(one) <= n else one[: n - 1].rsplit(" ", 1)[0] + "..."
