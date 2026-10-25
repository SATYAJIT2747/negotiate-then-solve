"""Typed tools the model calls, and the one place their calls are checked and run.

    @tool(Scope.READ)
    def classes_of(ctx: Context, args: ClassesOf) -> dict:
        '''The classes of a teacher, a section, a room or a course.'''

    registry = Registry([classes_of, ...])
    registry.specs({Scope.READ})                  # the "tools" list for the model, from the args models
    registry.run(call, ctx, allowed={Scope.READ})  # a ToolResult, never an exception

A handler that cannot do what was asked raises ``Refusal``; the model gets its code, message and hint
(``did_you_mean`` and the like) to fix the call. Arguments that do not fit the args model are refused
the same way, before the handler runs.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, get_type_hints

from pydantic import BaseModel, ValidationError

from nts.protocol import UNPARSED, ToolCall, ToolError, ToolResult
from nts.state import Draft, World


class Scope(str, Enum):
    READ = "read"  # looks at the state
    DRAFT = "draft"  # changes the episode's draft
    EVALUATE = "evaluate"  # asks the kernel about the draft
    TALK = "talk"  # sends a message to a person (the episode waits for the answer)
    POLICY = "policy"  # looks up the department's regulations
    END = "end"  # ends the episode: submit, decline, escalate


class Refusal(Exception):
    def __init__(self, code: str, message: str, **hint: Any) -> None:
        super().__init__(message)
        self.error = ToolError(code=code, message=message, hint=hint)


@dataclass
class Context:
    world: World  # the store's version when the episode started
    actor: str  # faculty id of the person asking, or "coordinator"
    week: int | None = None  # the current week of term; None outside term
    draft: Draft | None = None
    scratch: dict[str, Any] = field(default_factory=dict)  # what tools keep between calls in one episode
    handbook: Any = None  # the department's regulations (nts.policy.Handbook), for the rules tool
    request: str = ""  # what the person asked, as they wrote it

    @property
    def state(self) -> World:
        """What read tools look at: the draft if there is one, so the agent sees its own changes."""
        return self.draft.world if self.draft else self.world


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    scope: Scope
    args: type[BaseModel]
    handler: Callable[[Context, BaseModel], Any]

    def spec(self) -> dict:
        return {"type": "function",
                "function": {"name": self.name, "description": self.description,
                             "parameters": compact(self.args.model_json_schema())}}


def tool(scope: Scope, name: str | None = None) -> Callable[[Callable], Tool]:
    """Make a Tool of ``handler(ctx, args)``: the args model from its annotation, the description from its docstring."""

    def wrap(handler: Callable) -> Tool:
        params = list(inspect.signature(handler).parameters)
        args = get_type_hints(handler)[params[1]]
        return Tool(name or handler.__name__, inspect.cleandoc(handler.__doc__ or ""), scope, args, handler)

    return wrap


class Registry:
    def __init__(self, tools: Iterable[Tool]) -> None:
        self.tools: dict[str, Tool] = {}
        for t in tools:
            if t.name in self.tools:
                raise ValueError(f"two tools called {t.name}")
            self.tools[t.name] = t

    def specs(self, allowed: Iterable[Scope] | None = None) -> list[dict]:
        allowed = set(Scope) if allowed is None else set(allowed)
        return [t.spec() for t in self.tools.values() if t.scope in allowed]

    def run(self, call: ToolCall, ctx: Context, allowed: Iterable[Scope] | None = None) -> ToolResult:
        start = time.perf_counter()
        try:
            content = self._run(call, ctx, set(Scope) if allowed is None else set(allowed))
            return ToolResult(call_id=call.id, name=call.name, ok=True, content=content, ms=_ms(start))
        except Refusal as r:
            return ToolResult(call_id=call.id, name=call.name, ok=False, error=r.error, ms=_ms(start))
        except Exception as e:  # noqa: BLE001 -- a bug in a tool must not end the episode; telemetry keeps the type
            err = ToolError(code="internal", message=f"the tool failed ({type(e).__name__}); try another way")
            return ToolResult(call_id=call.id, name=call.name, ok=False, error=err, ms=_ms(start))

    def _run(self, call: ToolCall, ctx: Context, allowed: set[Scope]) -> Any:
        t = self.tools.get(call.name)
        if t is None or t.scope not in allowed:
            names = [n for n, x in self.tools.items() if x.scope in allowed]
            raise Refusal("unknown_tool", f"there is no tool {call.name!r}", tools=names)
        if UNPARSED in call.arguments:
            raise Refusal("bad_json", f"the arguments for {t.name} were not valid JSON")
        if unknown := sorted(set(call.arguments) - set(t.args.model_fields)):  # pydantic would drop them silently
            raise Refusal("bad_arguments", f"{t.name} has no field {', '.join(map(repr, unknown))}",
                          fields=list(t.args.model_fields))
        try:
            args = t.args.model_validate(call.arguments)
        except ValidationError as e:
            problems = [{"field": ".".join(map(str, x["loc"])) or "(arguments)", "problem": x["msg"]} for x in e.errors()]
            raise Refusal("bad_arguments", f"the arguments for {t.name} do not fit", problems=problems) from None
        return t.handler(ctx, args)


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)


def compact(schema: Any) -> Any:
    """Pydantic's JSON schema without what costs prompt tokens and tells the model nothing: "title" keys
    (they repeat the field names), ``anyOf [X, null]`` for optional fields (X: a field not in "required"
    may be left out) and ``null`` or empty defaults."""
    if isinstance(schema, list):
        return [compact(v) for v in schema]
    if not isinstance(schema, dict):
        return schema
    out = {}
    for k, v in schema.items():
        if k == "title" and isinstance(v, str):
            continue
        if k == "default" and v in (None, [], {}):
            continue
        if k == "anyOf" and isinstance(v, list) and {"type": "null"} in v and len(v) == 2:
            out.update(compact(next(x for x in v if x != {"type": "null"})))
            continue
        out[k] = compact(v)
    if "properties" in out and not out["properties"]:
        del out["properties"]
    return out
