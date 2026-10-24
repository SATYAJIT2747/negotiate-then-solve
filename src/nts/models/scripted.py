"""A model that plays back a script: for tests of the harness and tools, with no server.

    model = ScriptedModel([call("free_hours", teachers=["me"]), say("You are free on Monday at 9 am.")])

A step can also be a function of the conversation so far, to answer from what the tools returned.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence

from nts.protocol import AssistantTurn, Message, ToolCall

Step = AssistantTurn | Callable[[Sequence[Message]], AssistantTurn]
_ids = itertools.count(1)


def call(name: str, **arguments: object) -> AssistantTurn:
    return AssistantTurn(tool_calls=[ToolCall(id=f"call-{next(_ids)}", name=name, arguments=arguments)], model="scripted")


def say(text: str) -> AssistantTurn:
    return AssistantTurn(text=text, model="scripted")


class ScriptExhausted(RuntimeError):
    pass


class ScriptedModel:
    name = "scripted"

    def __init__(self, steps: Sequence[Step]) -> None:
        self.steps = list(steps)
        self.seen: list[list[Message]] = []  # the conversation each turn was asked with
        self.tools: list[list[dict]] = []

    def chat(self, messages: Sequence[Message], tools: Sequence[dict]) -> AssistantTurn:
        if len(self.seen) == len(self.steps):
            raise ScriptExhausted(f"the script has {len(self.steps)} turns")
        step = self.steps[len(self.seen)]
        self.seen.append(list(messages))
        self.tools.append(list(tools))
        return step(messages) if callable(step) else step
