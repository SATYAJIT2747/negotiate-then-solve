"""Offline parser: a rule-based stand-in for System Two, so the portal runs
without an API key. It reads a message into the same intents a model
returns (``read_intents``) and goes through the same deterministic
resolution and post-processing (``intents.compile_intents``: times, weeks,
classes, tiers, authority, validation), so every downstream layer behaves
as it would with the LLM. ``read`` and ``parse`` give the older
``ParseOutput`` format. It covers the common request shapes of the corpus;
anything it cannot read becomes a clarification rather than a guess.
"""

from __future__ import annotations

import re

from core.instance import Instance, SessionKind
from core.schemas import Request

from . import intents
from .corpus import INJECTIONS
from .intents import (
    Avoid,
    ClassRef,
    ExtraClass,
    IntentOutput,
    Move,
    NeedsRoom,
    Only,
    Period,
    RoomClosed,
    Swap,
    TimeSpec,
    Unavailable,
    Who,
    compile_intents,
)
from .parsing import ParseOutput, ParseResult

DAY_WORDS = {"mon": "Mon", "tue": "Tue", "wed": "Wed", "thu": "Thu", "fri": "Fri", "sat": "Sat", "sun": "Sun"}
_DAY = re.compile(r"\b(mon|tue|tues|wed|wednes|thu|thur|thurs|fri|sat|satur|sun)(?:day)?s?\b", re.IGNORECASE)
_RANGE = re.compile(r"\b(mon|tue|wed|thu|fri)\w*\s*(?:to|-|–|through|till|until)\s*(mon|tue|wed|thu|fri)\w*", re.IGNORECASE)
_WEEKS = re.compile(r"\bweeks?\s+(\d{1,2})(?:\s*(?:-|–|to|and)\s*(\d{1,2}))?", re.IGNORECASE)
_TIME = re.compile(r"\b(\d{1,2})(?::\d\d)?\s*(am|pm)\b", re.IGNORECASE)
_INJECTION = re.compile(r"(?im)^.*(ignore (all |previous )|system:|admin mode|assistant[,:]|approve all|"
                        r"publish (it |immediately|now)|override|</message>).*$")
UNAVAILABLE = re.compile(r"\b(conference|away|leave|unavailable|can't|cannot|can not|won't be|not be available|"
                         r"appointment|duty|out of station|travelling|traveling|not available)\b", re.IGNORECASE)
WISH = re.compile(r"\b(prefer|rather|if possible|would like|avoid|try to|keep .* free|where possible)\b", re.IGNORECASE)
ONLY = re.compile(r"\b(only|limit(ed)? to|restrict|keep my (lectures|classes|teaching) to|all my teaching on)\b", re.IGNORECASE)
AVOID = re.compile(r"\b(no classes|not teach|avoid|free|nothing before|don't schedule|do not schedule|rather not)\b", re.IGNORECASE)
QUESTION = re.compile(r"\b(rule|allowed|permitted|permissible|policy|who (has|needs) to approve|how many|"
                      r"can classes|is it ok)\b", re.IGNORECASE)
# "it has to be on Wednesday at 2 pm": where a class should be, not when the teacher is free
PLACE = re.compile(r"\b(must|has to|have to|needs? to|should|would like (it|them) to|want (it|them) to)\s+be\s+(on|at|in)\b",
                   re.IGNORECASE)
# "I want a 2 pm slot on Monday": a wish *for* a time; without it an unclassified time reads as one to avoid
WANT = re.compile(r"\b(i want|i'd like|i would like|i need|give me|can i (have|get)|could i (have|get)|"
                  r"(a|an) \S+ slot)\b", re.IGNORECASE)
# an extra hour the weekly timetable has no session for: the coordinator arranges it
EXTRA = re.compile(r"\b(extra|additional|revision|remedial|doubt.clearing)\s+(class|classes|lecture|lectures|session|sessions)\b",
                   re.IGNORECASE)
OUT_OF_SCOPE = re.compile(r"\b(auditorium|wifi|wi-fi|salary|projector|exam results|fest|book the)\b", re.IGNORECASE)
VAGUE = re.compile(r"\b(a few days|couple of days|soon|next month|later this semester|some days|fewer early)\b", re.IGNORECASE)
EQUIPMENT = {"gpu": "gpu", "graphics": "gpu", "router": "routers", "networking": "routers", "electronic": "electronics"}
SWAP = re.compile(r"\b(swap|swapp\w*|exchange|trade|switch)\b|"
                  r"\btake my\b.*\b(and )?I (take|teach|cover) (their|his|her)\b", re.IGNORECASE)
_TITLE_WORD = re.compile(r"^(dr|prof|professor|mr|ms|mrs)\.?$", re.IGNORECASE)


def surname(name: str) -> str:
    parts = [p for p in re.split(r"[\s.]+", name) if p and not _TITLE_WORD.match(p)]
    return parts[-1] if parts else name


def _slot(hour: int, ampm: str) -> int:
    h = hour % 12 + (12 if ampm.lower() == "pm" else 0)
    return h - 9


def _clock(hour: int, ampm: str) -> str:
    return f"{hour % 12 + (12 if ampm.lower() == 'pm' else 0):02d}:00"


class RuleParser:
    def __init__(self, instance: Instance) -> None:
        self.instance = instance
        cal = instance.calendar
        lunch = cal.lunch_slot if cal.lunch_slot is not None else cal.slots_per_day // 2
        self.morning = list(range(lunch))
        self.afternoon = list(range(lunch + 1, cal.slots_per_day))

    # -- reading the text ---------------------------------------------------------

    def days(self, text: str) -> list[str] | None:
        order = self.instance.calendar.days
        if m := _RANGE.search(text):
            a, b = DAY_WORDS[m[1][:3].lower()], DAY_WORDS[m[2][:3].lower()]
            if a in order and b in order and order.index(a) <= order.index(b):
                return order[order.index(a): order.index(b) + 1]
        found = [DAY_WORDS[d[:3].lower()] for d in _DAY.findall(text)]
        days = [d for d in order if d in found]
        return days or None

    def weeks(self, text: str) -> list[int] | None:
        if m := _WEEKS.search(text):
            a = int(m[1])
            b = int(m[2]) if m[2] else a
            return list(range(min(a, b), max(a, b) + 1))
        return None

    def times(self, text: str) -> TimeSpec:
        """The times of day the text names, days aside."""
        low = text.lower()
        if m := re.search(r"before\s+(\d{1,2})(?::\d\d)?\s*(am|pm)", low):
            return TimeSpec(before=_clock(int(m[1]), m[2]))
        if m := re.search(r"after\s+(\d{1,2})(?::\d\d)?\s*(am|pm)", low):
            return TimeSpec(after=_clock(int(m[1]), m[2]))
        # "move it before lunch as I won't be available after lunch": the first one is the ask, the rest the reason
        if m := re.search(r"\b(?:before (?:lunch|noon)|pre-lunch|morning)|\b(?:after lunch|afternoon)", low):
            return TimeSpec(part="afternoon" if m[0].startswith(("after", "afternoon")) else "morning")
        spd = self.instance.calendar.slots_per_day
        at = sorted({_clock(int(h), ap) for h, ap in _TIME.findall(low) if 0 <= _slot(int(h), ap) < spd})
        return TimeSpec(at=at) if at else TimeSpec()

    def slots(self, text: str) -> list[int] | None:
        try:
            return intents.slots(self.instance, self.times(text))
        except intents.Unresolved:
            return None

    def period(self, text: str) -> Period:
        return Period(weeks=self.weeks(text))

    def sessions_named(self, text: str, faculty: str | None) -> list[str]:
        low = text.lower()
        mine = [s for s in self.instance.sessions if faculty is None or s.faculty == faculty]
        named = [s for s in mine if self.instance.course_title(s.course).lower() in low]
        if re.search(r"\b(practical|lab session|lab)\b", low):
            named = [s for s in (named or mine) if s.kind == SessionKind.PRACTICAL]
        return [s.id for s in named]

    def other_faculty(self, text: str, sender: str) -> str | None:
        low = text.lower()
        for f in self.instance.faculty:
            if f.id != sender and f.name.lower() in low:
                return f.id
        return None

    def _names_room(self, low: str) -> bool:
        return any(r.name.lower() in low or r.id.lower() in low for r in self.instance.rooms)

    def mentioned(self, text: str, sender: str) -> list[str]:
        """Other faculty members the text names: by full name, else every one
        whose surname appears ("Dr. Rao" may fit two people)."""
        if found := self.other_faculty(text, sender):
            return [found]
        low = text.lower()
        return [f.id for f in self.instance.faculty
                if f.id != sender and re.search(rf"\b{re.escape(surname(f.name).lower())}\b", low)]

    def colleague(self, text: str, sender: str) -> str | None:
        hits = self.mentioned(text, sender)
        return hits[0] if len(hits) == 1 else None

    # -- parsing -----------------------------------------------------------------------

    def parse(self, request: Request) -> ParseResult:
        return compile_intents(self.instance, request, self.read_intents(request)).result

    def read(self, request: Request) -> ParseOutput:
        """The ``ParseOutput`` a model would return for ``request`` (before post-processing)."""
        return self.parse(request).output

    def read_intents(self, request: Request) -> IntentOutput:
        text = _INJECTION.sub("", request.raw_text)
        for inj in INJECTIONS:
            text = text.replace(inj, "")
        low = text.lower()
        sender = request.sender_id
        role = request.role.value

        if OUT_OF_SCOPE.search(low):
            return IntentOutput(action="out_of_scope")
        if role == "student" and re.search(r"\bclash|overlap|same time\b", low):
            return IntentOutput(action="investigate")
        if "?" in text and QUESTION.search(low) and not re.search(r"\b(could you|can you|please)\b", low):
            return IntentOutput(action="answer")

        days, period, times = self.days(text), self.period(text), self.times(text)
        weeks, slots = period.weeks, self.slots(text)

        def when(d: list[str] | None = days, t: TimeSpec = times) -> TimeSpec:
            return t.model_copy(update={"days": d})

        if role != "student" and EXTRA.search(low) and not UNAVAILABLE.search(low):
            return self._compile(ExtraClass(which=self._class(text, sender), period=period, time=when()))

        if role != "student" and SWAP.search(low) and (people := self.mentioned(text, sender)):
            # which two sessions is decided against the timetable (agents.swap), not here
            theirs = ClassRef(owner=people[0]) if len(people) == 1 else ClassRef()
            return self._compile(Swap(mine=ClassRef(), theirs=theirs, period=period))

        other = self.other_faculty(text, sender)
        who = Who(kind="faculty", id=other) if other else None

        # rooms: a lab in-charge reporting an outage, or the exam cell taking a room for exams
        if role in ("lab_incharge", "exam_cell") and (role == "lab_incharge" or self._names_room(low)):
            lab = next((r for r in self.instance.rooms if r.name.lower() in low or r.id.lower() in low), None)
            if lab is None:
                return self._clarify("room_closed", ["room"], "Which room is affected?")
            if weeks is None and days is None:
                return self._clarify("room_closed", ["weeks"], f"Which weeks is {lab.name} unavailable?")
            return self._compile(RoomClosed(room=lab.id, time=TimeSpec(days=days), period=period))

        # equipment a session needs
        needs = sorted({v for k, v in EQUIPMENT.items() if k in low})
        if needs:
            sessions = self.sessions_named(text, None if other else sender)
            if other:
                sessions = [s.id for s in self.instance.sessions if s.faculty == other and
                            s.kind == SessionKind.PRACTICAL] or sessions
            if not sessions:
                return self._clarify("needs_room", ["session"], "Which of your classes needs this equipment?")
            which = ClassRef(session=sessions[0])
            found = [NeedsRoom(which=which, equipment=needs, justification="stated")]
            if days or slots:
                found.append(Move(which=which, to=when(), period=period, hard=True))
            return self._compile(*found)

        if (VAGUE.search(low) and not days and not weeks) or (UNAVAILABLE.search(low) and not days and not weeks):
            missing = [m for m, v in (("days", days), ("weeks", weeks), ("slots", slots)) if v is None][:2]
            return self._clarify("unavailable" if UNAVAILABLE.search(low) else "avoid", missing or ["days"],
                                 "Could you tell me which " + " and ".join(missing or ["days"]) + " you mean?")

        if UNAVAILABLE.search(low):
            # "can't on Mon, Wed or Fri; I am only on campus on Tue and Thu": the days they *can* come are not
            # unavailable ones. Take the days from the clauses that say when they cannot, when those name days.
            clauses = re.split(r"[;.]|\bbut\b", text)
            off = [c for c in clauses if UNAVAILABLE.search(c)]
            if any(ONLY.search(c) and not UNAVAILABLE.search(c) for c in clauses) and (d := self.days(" ".join(off))):
                days = d
            stated = re.search(r"\b(because|due to|conference|appointment|duty)\b", low)
            return self._compile(Unavailable(who=who, time=when(days), period=period,
                                             justification="stated" if stated else "none"))

        # a specific placement ("schedule my lecture at 1 pm on Friday", "4 lectures back-to-back")
        if re.search(r"\b(schedule|move|put|shift|stack)\b", low) and (days or slots):
            sessions = self.sessions_named(text, other or sender)
            if re.search(r"back.to.back|in a row|straight|consecutive|without (any )?breaks?", low):
                t = times if slots else TimeSpec(part="morning")
                return self._compile(Only(who=who, time=when(t=t), hard=True))
            if sessions:
                return self._compile(Move(which=ClassRef(session=sessions[0]), to=when(), period=period, hard=True))
            return self._compile(Only(who=who, time=when(), period=period, hard=True))

        if days or slots:
            hard = not WISH.search(low) and bool(re.search(r"\b(must|need|has to|have to|cannot)\b", low))
            if PLACE.search(low) and not AVOID.search(low):
                sessions = self.sessions_named(text, other or sender)
                if sessions:
                    return self._compile(Move(which=ClassRef(session=sessions[0]), to=when(), period=period,
                                              hard=hard))
                return self._compile(Only(who=who, time=when(), period=period, hard=hard))
            if (ONLY.search(low) or WANT.search(low)) and days and not AVOID.search(low):
                return self._compile(Only(who=who, time=when(), period=period, hard=hard))
            return self._compile(Avoid(who=who, time=when(), period=period, hard=hard))

        return self._clarify("avoid", ["days", "slots"], "Could you tell me which days and times you mean?")

    def _class(self, text: str, sender: str) -> ClassRef:
        """The sender's class a message names: course and kind, when it names them."""
        low = text.lower()
        named = {self.instance.session_by_id[s].course for s in self.sessions_named(text, sender)}
        kind = next((k for k, pat in (("practical", r"\b(practical|lab session|lab)\b"), ("tutorial", r"\btutorials?\b"))
                     if re.search(pat, low)), "lecture")
        return ClassRef(course=named.pop() if len(named) == 1 else None, kind=kind)

    @staticmethod
    def _compile(*found) -> IntentOutput:
        return IntentOutput(action="compile", intents=list(found))

    @staticmethod
    def _clarify(topic: str, missing: list[str], question: str) -> IntentOutput:
        return IntentOutput(action="clarify", topic=topic, missing=missing, question=question)
