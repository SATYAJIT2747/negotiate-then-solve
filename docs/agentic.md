# The agent-loop architecture (`src/nts`)

A rebuild of Negotiate, Then Solve around one loop: the model reads the department's state through
tools, makes changes in a private draft, asks the kernel to check them, and submits. The old packages
(`core`, `language`, `agents`, `pipeline`, ...) stay as they are, as the baseline the new design is
compared against.

## The rule everything rests on

The model never writes the timetable. Only `Store.commit` changes the state, and it re-checks every
change with the kernel's validator whatever the agent claimed. So the CP-SAT guarantee holds, and
every model mistake becomes an error the model sees (and a training example), not a bad timetable.

## Layers

| Package | What it is | Status |
|---|---|---|
| `nts.kernel` | `core/` and the concession ledger, copied unchanged: instance, CP-SAT solver, validator | done |
| `nts.state` | append-only event log (hash chain), immutable `World` versions, drafts, the commit gate | done |
| `nts.tools` | typed tools (pydantic args -> JSON schema), scopes, `Refusal` errors with hints, name resolution | read tools done |
| `nts.protocol` | messages, tool calls, results: what every layer passes and telemetry records | done |
| `nts.models` | one `chat(messages, tools)` interface: scripted (tests), then replay, llama.cpp, Gemini | scripted done |
| `nts.harness` | the loop: turn budgets, policy gate, repeat detection, context trimming, pause/resume | next |
| `nts.telemetry` | spans episode > turn > model_call / tool_call; JSONL per day + SQLite index | next |
| `nts.data` | labels (validator, solver, approval, judge), SFT / DPO exporters, frozen splits, replay | later |
| `nts.eval` | scenarios = start state + request + checks on the committed state; paired runs vs the old pipeline | later |
| `nts.api` | FastAPI + server-sent events of each turn | later |

## Tools, by scope

- **read**: `find`, `describe`, `classes_of`, `free_hours`, `who_is_free`, `free_rooms`, `changes` (done)
- **draft**: `propose(intent)`, `retract`, `list_draft`
- **evaluate**: `check_draft` -> feasible or not, the diff, who is affected, burden; conflicts in words
- **talk**: `ask_requester`, `message_party`, `make_offer` (the episode waits for the answer)
- **end**: `submit`, `decline(reason)`, `escalate(brief)`

Tools take and return the department's names ("Dr. Iyer", "Lab 2", "Tue", "2 pm"), never ids. A name that
matches nothing is refused with `did_you_mean`, so the model fixes the call on its next turn.

## Build order

1. Kernel, state store, read tools, scripted model, tests. **(done)**
2. Harness loop + budgets, telemetry, replay model: every test deterministic.
3. Draft and evaluate tools; the coordinator end to end on Gemini (clean first traces).
4. Exporters; a 4B LoRA on the traces; the same scenarios on the local model.
5. Talk tools, negotiation with simulators, API and UI.

## Running the tests

    uv run pytest tests/nts        # or, offline: ../negotiate-then-solve/.venv/Scripts/python -m pytest tests/nts
