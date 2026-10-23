"""The model's lookups (agents.tool_agent), with a fake model: what the code makes of a choice, what it sends
back when the choice is wrong, and the limit on tries. No API calls."""

from datetime import date

import pytest

from agents.tool_agent import MAX_TRIES, Choice, ToolAgent
from web.app import World

WEEK = 13


class FakeModel:
    """Answers with the given choices in turn, and keeps the prompts it was sent."""

    def __init__(self, *choices: Choice) -> None:
        self.choices = list(choices)
        self.prompts: list[str] = []

    def generate(self, system: str, prompt: str, schema: type[Choice]) -> Choice:
        self.prompts.append(prompt)
        return self.choices.pop(0)


@pytest.fixture(scope="module")
def world():
    return World(parser_mode="offline", seed_history=False, deadline=60)


def person(world, name):
    return next(f.id for f in world.instance.faculty if f.name == name)


def agent(world, *choices, sender="Dr. Menon"):
    who = person(world, sender) if sender.startswith("Dr.") else sender
    return ToolAgent(world.instance, FakeModel(*choices), who, WEEK, today=date(2026, 10, 21))


def test_names_become_ids_and_me_is_the_sender(world):
    a = agent(world, Choice(tool="free_hours", teachers=["me", "Dr. Iyer"], days=["Fri"], week=WEEK))
    call = a.read("when are iyer and I both around on friday?")
    assert call.tool == "free_hours" and call.week == WEEK and call.confident
    assert call.args == {"teachers": [person(world, "Dr. Menon"), person(world, "Dr. Iyer")], "days": ["Fri"]}


def test_the_prompt_has_the_names_and_the_question_but_no_timetable(world):
    a = agent(world, Choice(tool="none", week=WEEK))
    a.read("when's Iyer around this week?")
    prompt = a.client.prompts[0]
    assert "Dr. Iyer" in prompt and "Lab 2 (lab, 60 seats, gpu, routers)" in prompt and "week 13" in prompt
    assert "<question>when's Iyer around this week?</question>" in prompt
    assert "C-0" not in prompt and "Room 101 (lecture" in prompt  # names only: no IDs, no classes


def test_none_is_not_a_timetable_question(world):
    assert agent(world, Choice(tool="none", week=WEEK)).read("can I have Friday off?") is None


def test_a_wrong_name_is_sent_back_and_corrected(world):
    a = agent(world, Choice(tool="classes_of", teachers=["Dr. Iyar"], week=WEEK),
              Choice(tool="classes_of", teachers=["Dr. Iyer"], week=WEEK))
    call = a.read("what does iyar teach")
    assert call.args == {"teacher": person(world, "Dr. Iyer")}
    assert 'no teacher is called "Dr. Iyar"' in a.client.prompts[1]
    assert [t["problem"] is None for t in a.trace] == [False, True]


def test_after_the_last_try_it_says_so_and_guesses_nothing(world):
    wrong = Choice(tool="free_hours", days=["Sat"], teachers=["me"], week=WEEK)
    a = agent(world, *[wrong] * MAX_TRIES)
    call = a.read("am I free on saturday")
    assert not call.confident and "couldn't work out" in call.note
    assert len(a.client.prompts) == MAX_TRIES and all(t["problem"] for t in a.trace)


@pytest.mark.parametrize("choice, problem", [
    (Choice(tool="free_hours", week=WEEK), "needs at least one teacher"),
    (Choice(tool="free_rooms", week=WEEK), "needs kind"),
    (Choice(tool="free_rooms", kind="lab", equipment=["laser"], week=WEEK), "equipment ['laser']"),
    (Choice(tool="free_hours", teachers=["me"], hours=[4], week=WEEK), "not lunch"),
    (Choice(tool="classes_of", teachers=["Dr. Iyer", "Dr. Rao"], week=WEEK), "at most one teacher"),
    (Choice(tool="free_hours", teachers=["me"], week=40), "week 40 does not exist"),
])
def test_what_is_checked(world, choice, problem):
    a = agent(world, *[choice] * MAX_TRIES)
    a.read("question")
    assert problem in a.trace[0]["problem"]


def test_something_not_in_the_department_is_not_guessed(world):
    call = agent(world, Choice(tool="free_hours", unknown=["Dr. Smith"], week=WEEK)).read("is smith free?")
    assert not call.confident and "Dr. Smith" in call.note
    call = agent(world, Choice(tool="free_rooms", kind="lecture", unknown=["auditorium"], week=WEEK)).read("auditorium free?")
    assert not call.confident and "auditorium" in call.note


def test_a_class_representatives_me_is_their_section(world):
    call = agent(world, Choice(tool="free_hours", teachers=["me", "Dr. Iyer"], week=WEEK), sender="ST-G-04").read(
        "when are we and iyer both free")
    assert call.args == {"teachers": [person(world, "Dr. Iyer")], "sections": ["G-04"]}


def test_free_rooms_and_classes_of_take_the_regex_shapes(world):
    call = agent(world, Choice(tool="free_rooms", kind="lab", equipment=["gpu"], days=["Tue"], hours=[6, 5],
                               week=WEEK + 1)).read("any gpu lab free next tuesday afternoon 2 or 3?")
    assert call.args == {"kind": "lab", "equipment": ["gpu"], "days": ["Tue"], "slots": [5, 6]} and call.week == WEEK + 1
    call = agent(world, Choice(tool="classes_of", course="ML", week=WEEK)).read("when's ML?")
    assert call.args == {"course": next(c for c, t in world.instance.course_titles.items() if t == "Machine Learning")}
