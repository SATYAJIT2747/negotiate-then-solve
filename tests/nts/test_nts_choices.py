"""Answers to pick, read off the agent's question (nts.api.choices)."""

import pytest

from nts.api.choices import choices


@pytest.mark.parametrize(("question", "picks"), [
    ("Wednesday 9 am, 10 am, or 11 am is free. Would any of those work?",
     ["Wednesday 9 am", "Wednesday 10 am", "Wednesday 11 am", "None of these"]),
    ("Is Tuesday at 2 pm or Thursday at 4 pm better for you?", ["Tuesday 2 pm", "Thursday 4 pm", "None of these"]),
    ("Which kind of class (lecture or practical) would you like to add?", ["lecture", "practical"]),
    ("Do you mean the Monday lecture or the Friday lab?", ["Monday lecture", "Friday lab", "None of these"]),
    ("You mentioned Tuesday at 2 pm, but that time is taken. Would another time on Tuesday work, or would you prefer "
     "a different day?", ["Another time on Tuesday", "A different day"]),
    ("Should I move it now, or wait until next week?", ["Move it now", "Wait until next week"]),
    ("Shall I send it to the office?", ["Yes", "No"]),
    ("Which week should this apply to?", []),
])
def test_choices(question, picks):
    assert choices(question) == picks
