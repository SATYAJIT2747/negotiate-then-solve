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
* replies and notices are built from constraints, never from private reasons.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime

from agents.explainer import describe, placement_text, session_name
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
from agents.negotiation import Message, Negotiator, Offer, Outcome, Responder
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
    Constraint,
    ConstraintType,
    Placement,
    Request,
    RequestStatus,
    Role,
    Scope,
    Tier,
    TimetableVersion,
    When,
)
from core.semantics import PLACEMENT_TYPES, room_compatible, sessions_in_scope, violates
from core.solver import TimetableSolver
from core.validator import validate_constraint, verify_timetable
from language.corpus import FULL_DAY, ExpectedAction, RequestType, hour
from language.intents import Cancel as CancelIntent
from language.intents import Compiled
from language.intents import ExtraClass as ExtraIntent
from language.intents import Swap as SwapIntent
from language.parsing import ParseResult
from language.rule_parser import RuleParser

from .store import EXTRA_PREFIX, Store

S = RequestStatus
CHOICES = 3  # times offered to a sender whose request fits in more than one way


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

    def intents(self, kind: type) -> list:
        """The parsed intents of type ``kind`` (none for a parser without intents)."""
        out = self.compiled.output if self.compiled is not None else None
        return [i for i in out.intents if isinstance(i, kind)] if out is not None and out.action == "compile" else []

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
    owed) rather than left in a closed room or the closure escalated."""
    if week is None:
        return set(), []
    closed = {c.scope.room for c in constraints
              if c.hard and c.type == ConstraintType.UNAVAILABLE and c.scope.room and c.active_in(week)
              and (c.when.weeks is None or week in c.when.weeks) and not c.when.days and not c.when.slots}
    if not closed:
        return set(), []
    out = set()
    for s in instance.sessions:
        rooms = [r.id for r in instance.rooms if room_compatible(instance, s, r)]
        if rooms and set(rooms) <= closed:
            out.add(s.id)
    return out, sorted(instance.room_by_id[r].name for r in closed)


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
        policy_agent: PolicyAgent | None = None,
        responders: Mapping[str, Responder] | None = None,
        approvers: Iterable[str] = ("C-TT",),
        semester: str = "current",
        negotiator_factory: Callable[[], Negotiator] | None = None,
        swap_reader=None,
        offer_choices: Callable[[], bool] | None = None,
        current_week: Callable[[], int | None] | None = None,
    ) -> None:
        """``offer_choices``: whether, right now, a sender whose request fits in more than one way is
        asked which time they prefer (``_let_sender_choose``); never when omitted. ``current_week``: the
        teaching week now, so a parser reading intents can resolve "next week"; unknown when omitted.

        A parser with ``compile(request, timetable, current_week)`` reads intents (``language.intents``):
        swaps, extra classes, cancellations and withdrawals are routed by what it read. ``compile`` may
        return None (a stand-in without intents answers): the request is then parsed as before."""
        self.instance = instance
        self.store = store
        self.parser = parser
        self.fast_parser = fast_parser
        self.system_one = system_one
        self.tau = tau
        self.policy = policy_agent
        self.negotiator = negotiator
        self.responders = dict(responders or {})
        self.approvers = set(approvers)
        self.semester = semester
        self.negotiator_factory = negotiator_factory
        self.swap_reader = swap_reader or RuleSwapReader(instance)
        self.extra_reader = ExtraReader(instance)
        self.offer_choices = offer_choices
        self.current_week = current_week
        self.cases: dict[str, Case] = {}

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

    # -- the lifecycle ---------------------------------------------------------------

    def _advance(self, case: Case, status: RequestStatus, **info) -> None:
        case.request.advance(status)
        self.store.save_request(case.request)
        self.store.log(case.id, status.value, **info)

    def _parse(self, case: Case) -> ParseResult:
        r = case.request
        if self.system_one is not None:
            d = self.system_one.decide(r)
            # plain Python values: numpy scalars from System One cannot be sent as JSON
            fast = bool(d.fast_path(self.tau)) and self.fast_parser is not None
            case.routing = {"request_type": d.request_type.value, "type_p": round(float(d.type_p), 4),
                            "action": d.action.value, "action_p": round(float(d.action_p), 4),
                            "tau": float(self.tau), "fast_path": fast, "latency_ms": round(float(d.latency_ms), 2)}
            if fast:
                case.route = "fast"
                return self._read(case, self.fast_parser)
        case.route = "system_two"
        return self._read(case, self.parser)

    def _read(self, case: Case, parser) -> ParseResult:
        """Intents resolved against the published timetable when ``parser`` reads them, else the plain parse."""
        compile_ = getattr(parser, "compile", None)
        if compile_ is not None:
            base = self.store.current_version()
            week = self.current_week() if self.current_week else None
            case.compiled = compile_(case.request, _regular(base.assignment) if base else {}, week)
            if case.compiled is not None:
                case.route += "+intents"
                return case.compiled.result
        return parser.parse(case.request)

    def submit(self, request: Request) -> Case:
        case = Case(request=request)
        self.cases[case.id] = case
        if self.store.request(request.id) is None:
            self.store.add_request(request)
        self.store.log(case.id, "received", channel=request.channel.value, sender=request.sender_id)

        t = time.perf_counter()
        case.parse = self._parse(case)
        case.timings["parse"] = time.perf_counter() - t
        self._advance(case, S.CLASSIFIED, route=case.route, action=case.parse.action.value,
                      routing=case.routing, seconds=round(case.timings["parse"], 3))
        action = case.parse.action

        if action == ExpectedAction.WITHDRAW:
            return self._withdraw_by_message(case)
        if self._is_swap(case):
            return self._swap(case)
        if self._is_extra(case):
            return self._extra(case)
        if case.intents(CancelIntent) and action != ExpectedAction.REFUSE:
            return self._cancel(case)
        if action == ExpectedAction.REFUSE:
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
        if action in (ExpectedAction.INVESTIGATE, ExpectedAction.OUT_OF_SCOPE):
            to = "coordinator" if action == ExpectedAction.INVESTIGATE else "the relevant office"
            case.reply = f"Thank you. Your message has been passed to {to}."
            self._advance(case, S.FORWARDED, to=to)
            return case

        # compile or clarify: check policy first
        if self.policy is not None and action == ExpectedAction.COMPILE:
            t = time.perf_counter()
            case.policy = self.policy.review(request.raw_text, case.parse)
            case.timings["policy"] = time.perf_counter() - t
        self._advance(case, S.POLICY_CHECKED,
                      verdict=case.policy.verdict.value if case.policy else "not_checked")
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
            case.reply = f"Your request needs approval and has been sent up: {case.policy.explanation}"
            self._advance(case, S.ESCALATED, reason="needs approval", cited=case.policy.cited)
            return case

        case.constraints = (case.parse.constraints if case.compiled is not None else
                            self._on_named_day(case, case.parse.constraints))
        case.superseded = self._supersede(case)
        self.store.add_constraints(case.constraints, request_id=case.id)
        self._advance(case, S.COMPILED, constraints=[c.id for c in case.constraints],
                      obligations=case.policy.obligations if case.policy else [], superseded=case.superseded)
        return self._solve(case)

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
        if plan and self.policy is not None:
            t = time.perf_counter()
            view = ParseResult(request=r, output=case.parse.output, constraints=pins)
            case.policy = self.policy.review(r.raw_text, view)
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
            case.reply = f"{who} {why}, so the swap was not made. A swap needs the consent of both of you."
            self._advance(case, S.DENIED, reason=f"swap declined by {plan.counterpart}", cited=["P-SWAP"])
            return case

        case.constraints = pins
        superseded = self._supersede(case)
        self.store.add_constraints(case.constraints, request_id=case.id)
        self._advance(case, S.COMPILED, constraints=[c.id for c in pins], superseded=superseded,
                      consent=plan.counterpart)
        return self._solve(case)

    # -- cancellations and withdrawals in a message (intent parsers) -------------------------

    def _cancel(self, case: Case) -> Case:
        """A class not held in some weeks: the coordinator arranges it (and the make-up), so the request
        goes to them with the class and weeks resolved, or back to the sender when those are unclear."""
        c = case.compiled
        case.route += "+cancel"
        if c.question is not None:
            self._advance(case, S.POLICY_CHECKED, verdict="not_checked")
            case.reply = c.question
            self._advance(case, S.CLARIFICATION, missing=c.missing)
            return case
        base = self.store.current_version()
        placed = base.assignment if base else {}
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

    def _withdraw_by_message(self, case: Case) -> Case:
        """"Please ignore my earlier request": the sender's one open request is taken back. With several
        open, nothing is withdrawn on a guess: they are listed for the sender to choose."""
        sender = case.request.sender_id
        case.route += "+withdraw"
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
        if reading.template is not None and self.policy is not None:
            t = time.perf_counter()
            case.policy = self.policy.review(r.raw_text, ParseResult(request=r, output=case.parse.output))
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
            case.reply = f"Your request needs approval and has been sent up: {case.policy.explanation}"
            self._advance(case, S.ESCALATED, reason="needs approval", cited=case.policy.cited)
            return case
        return self._place_extra(case)

    def _place_extra(self, case: Case) -> Case:
        """Find a time in the week when the teacher, the section and a room are free, and propose the
        week's timetable with the extra class in it. Nothing else moves, so there is nothing to negotiate."""
        r = case.request
        reading = self._extra_reading(case)
        week, label = reading.week, self.extra_reader.label(reading.template)
        session = extra_session(reading.template, case.id)
        base = self.store.current_version(week) or self.store.current_version()
        assignment = _regular(base.assignment) if base else {}
        cancelled, _ = closures(self.instance, self.store.constraints(), week)
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
        case.proposal = self.store.propose_version(assignment, case.id, week=week, cancelled=sorted(cancelled))
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
        weeks = sorted({w for c in case.constraints for w in (c.when.weeks or [])})
        week = weeks[0] if weeks else None
        base = self.store.current_version(week) or self.store.current_version()
        baseline = _regular(base.assignment) if base else None
        ledger = ConcessionLedger(self.store.ledger())
        neg = self.negotiator_factory() if self.negotiator_factory else self.negotiator
        neg.ledger = ledger
        neg.priority.ledger = ledger
        neg.priority.baseline = dict(baseline or {})
        cancelled, closed = closures(self.instance, self.store.constraints(), week)
        neg.skip = frozenset(cancelled)
        if base is not None and set(cancelled) == set(base.cancelled) and self._already_met(case, base, week):
            # checked directly, not by the solver: a solve cut short by its time limit could move
            # classes nobody asked to move
            case.reply = ("Your request already fits the timetable, so nothing needed to move. "
                          "It is recorded, and later changes will keep to it.")
            self._advance(case, S.SOLVED, feasible=True, already_met=True)
            self._advance(case, S.FAIRNESS_AUDITED)
            self._advance(case, S.PUBLISHED, version=None, moved=[])
            return case

        def on_event(kind: str, data: dict) -> None:
            if kind == "message" and case.status == S.COMPILED:
                self._advance(case, S.SOLVED, feasible=False)
                self._advance(case, S.NEGOTIATING)
            self.store.log(case.id, f"negotiation_{kind}", **data)

        neg.listener = on_event
        t = time.perf_counter()
        outcome = neg.resolve(self.store.constraints(), self.responders, baseline=baseline, week=week)
        case.timings["solve_and_negotiate"] = time.perf_counter() - t
        case.outcome = outcome
        if case.status == S.COMPILED:
            self._advance(case, S.SOLVED, feasible=outcome.status == "feasible",
                          solver_seconds=round(outcome.result.wall_time, 3) if outcome.result else None)
        if outcome.status != "feasible" and case.status == S.SOLVED:
            self._advance(case, S.NEGOTIATING, rounds=outcome.rounds)
        if outcome.status == "escalated":
            # until someone decides, the request is not in force: later requests must not run into it
            self._withdraw(case)
            case.reply = "Your request conflicts with others and has been escalated for a decision."
            self._advance(case, S.ESCALATED, to=outcome.escalation.to, reason=outcome.escalation.reason,
                          brief=outcome.escalation.text)
            return case

        for cid in outcome.relaxed:
            self.store.set_active(cid, False)
        known = {c.id for c in self.store.constraints(active_only=False)}
        self.store.add_constraints([c for c in outcome.constraints if c.id not in known], request_id=case.id)
        for e in outcome.concessions:
            self.store.record_concession(e)
        case.fairness = self._audit(outcome)
        self._advance(case, S.FAIRNESS_AUDITED, **case.fairness)

        assignment = outcome.result.assignment
        moved = moved_sessions(baseline or {}, assignment)
        chosen = None
        if outcome.status == "feasible" and moved and not cancelled:
            assignment, chosen = self._let_sender_choose(case, assignment, baseline, week, neg)
            moved = moved_sessions(baseline or {}, assignment)
        if not moved and not cancelled and not outcome.notices.get(case.request.sender_id):
            # the timetable already meets it: nothing to approve or publish, but it stays in force
            case.reply = ("Your request already fits the timetable, so nothing needed to move. "
                          "It is recorded, and later changes will keep to it.")
            self._advance(case, S.PUBLISHED, version=None, moved=[])
            return case
        case.proposal = self.store.propose_version(assignment, case.id, week=week, cancelled=sorted(cancelled))
        case.reply = ("Your request can be met. The change is waiting for the coordinator's approval."
                      if not outcome.notices.get(case.request.sender_id) else
                      " ".join(outcome.notices[case.request.sender_id]))
        if chosen:
            case.reply = f"{chosen} The change is waiting for the coordinator's approval."
        if cancelled:
            others = len(set(moved) - cancelled)
            case.reply = (f"Recorded: {' and '.join(closed)} closed in week {week}. "
                          f"{len(cancelled)} {'class has' if len(cancelled) == 1 else 'classes have'} no other room "
                          "that week and will be cancelled, with a make-up owed"
                          + (f"; {others} other {'class moves' if others == 1 else 'classes move'}" if others else "")
                          + ". Their teachers and sections are told once the timetable office approves the change.")
        self._advance(case, S.AWAITING_APPROVAL, version=case.proposal.version, moved=moved)
        return case

    def _let_sender_choose(self, case: Case, assignment: Mapping[str, Placement], baseline, week: int | None,
                           neg: Negotiator) -> tuple[Mapping[str, Placement], str | None]:
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
                                  seed=neg.seed, skip=neg.skip).solve()
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

    def _audit(self, outcome: Outcome) -> dict:
        teaching = sorted({s.faculty for s in self.instance.sessions})
        ledger = ConcessionLedger(self.store.ledger())
        before = ConcessionLedger([e for e in ledger.entries if e not in outcome.concessions])
        return {"gini_before": round(before.gini(teaching, self.semester), 4),
                "gini_after": round(ledger.gini(teaching, self.semester), 4),
                "conceded": sorted({e.stakeholder for e in outcome.concessions}),
                "preferences_lost": sorted(outcome.notices)}

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
        if case.extra is not None:
            return None  # placed when granted
        if case.parse is None or not case.parse.constraints:
            return "this request has nothing to put into the timetable"
        if case.outcome is not None and not self.overrides(case):
            return ("it cannot fit even on its own, so granting would not help; "
                    "decline it, or ask the sender for a narrower request")
        return None

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

    def handle(self, case_id: str, handler: str, note: str, name: str | None = None) -> Case:
        """The coordinator answers a message forwarded to them; the sender gets the answer as the reply.
        The request stays forwarded: nothing in it reaches the solver."""
        case = self.cases[case_id]
        if case.forwarded_to != "coordinator":
            raise ValueError(f"case {case_id} was not forwarded to the coordinator ({case.status.value})")
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
        # people who gave something up for it hear that it no longer stands
        case.notices = {p: f"{case.id} was withdrawn by its sender; what you agreed to for it no longer applies."
                        for p in sorted(set(refunded))}
        case.reply = "You withdrew this request. The timetable is as it was before it."
        self._advance(case, S.WITHDRAWN, by=by, constraints=sorted(own), restored=restored,
                      refunded=sorted(set(refunded)), version=case.proposal.version if case.proposal else None)
        return case

    # -- human approval (L6) -------------------------------------------------------------

    def pending(self) -> list[Case]:
        return [c for c in self.cases.values() if c.status == S.AWAITING_APPROVAL]

    def approve(self, case_id: str, approver: str) -> TimetableVersion:
        if approver not in self.approvers:
            self.store.log(case_id, "approval_refused", approver=approver)
            raise NotAuthorised(f"{approver} may not approve timetable changes")
        case = self.cases[case_id]
        if case.status != S.AWAITING_APPROVAL or case.proposal is None:
            raise ValueError(f"case {case_id} is not awaiting approval ({case.status.value})")
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
        for person, text in self._cancel_notices(v).items():
            case.notices[person] = f"{case.notices[person]} {text}" if person in case.notices else text
        self._advance(case, S.PUBLISHED, version=v.version, approver=approver, notified=sorted(case.notices))
        if v.week is None:
            self.carry_into_weeks(v, approver)
        return v

    def carry_into_weeks(self, v: TimetableVersion, approver: str) -> list[TimetableVersion]:
        """A week with its own repair (an absence in week 7) shows that repair, not the semester
        timetable; a semester change published later must reach it too. Each such week is solved
        again on top of ``v`` with that week's constraints, and published with ``v``'s approval."""
        out = []
        weeks = sorted({x["week"] for x in self.store.versions() if x["published"] and x["week"] is not None})
        for week in weeks:
            cancelled, _ = closures(self.instance, self.store.constraints(), week)
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

    def _rebase(self, case: Case, base: TimetableVersion) -> TimetableVersion:
        """Another change was published after this proposal was made. Publishing the proposal as it is
        would put back the timetable it was built on and undo that change, so solve it again from the
        live version, with every constraint in force (this request's included)."""
        week = case.proposal.week
        cancelled, _ = closures(self.instance, self.store.constraints(), week)
        result = TimetableSolver(self.instance, self.store.constraints(), week=week,
                                 baseline=_regular(base.assignment), time_limit=30, skip=cancelled).solve()
        if not result.ok:
            raise ValueError("the timetable has changed since this proposal and it no longer fits; "
                             "reject it and ask the requester to send it again")
        v = self.store.propose_version(result.assignment, case.id, week=week, cancelled=sorted(cancelled))
        self.store.log(case.id, "rebased", stale=case.proposal.version, on=base.version, version=v.version)
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

    def _cancel_notices(self, v: TimetableVersion) -> dict[str, str]:
        """Teachers and sections whose class is not held in ``v``'s week, and why."""
        if not v.cancelled:
            return {}
        _, closed = closures(self.instance, self.store.constraints(), v.week)
        where = " and ".join(closed) or "its room"
        out: dict[str, list[str]] = {}
        for sid in v.cancelled:
            s = self.instance.session_by_id[sid]
            for person in [s.faculty, *(f"ST-{g}" for g in s.groups)]:
                out.setdefault(person, []).append(session_name(self.instance, sid))
        return {person: f"Week {v.week}: {', '.join(names)} {'is' if len(names) == 1 else 'are'} cancelled, "
                        f"because {where} is closed and no other room has what it needs. A make-up class is owed."
                for person, names in out.items()}

    def reject(self, case_id: str, approver: str, reason: str = "") -> None:
        if approver not in self.approvers:
            raise NotAuthorised(f"{approver} may not review timetable changes")
        case = self.cases[case_id]
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
                    f"{session_name(self.instance, s)} is now on {placement_text(self.instance, v.assignment[s])}"
                    for s in mine)
        return out


def _regular(assignment: Mapping[str, Placement]) -> dict[str, Placement]:
    """An assignment without its extra classes (the solver and the validator know only weekly sessions)."""
    return {s: p for s, p in assignment.items() if not s.startswith(EXTRA_PREFIX)}


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


def _short(text: str, n: int = 60) -> str:
    """A message's first words, for naming a request in a reply."""
    one = " ".join(text.split())
    return one if len(one) <= n else one[: n - 1].rsplit(" ", 1)[0] + "..."
