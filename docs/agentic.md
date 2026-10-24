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
| `nts.tools` | typed tools (pydantic args -> JSON schema), scopes, `Refusal` errors with hints, name resolution | read, draft, evaluate, end done |
| `nts.protocol` | messages, tool calls, results: what every layer passes and telemetry records | done |
| `nts.models` | one `chat(messages, tools)` interface: scripted (tests), replay, llama.cpp (`OpenAIChat`, LoRA per request), Gemini | done |
| `nts.harness` | the loop: budgets (turns, calls, tokens, time), repeat refusal, context trimming; pause/resume later | done |
| `nts.telemetry` | spans episode > turn > model_call / tool_call; JSONL per day + SQLite index | done |
| `nts.data` | labels (validator, solver, approval, judge), SFT / DPO exporters, frozen splits, replay | later |
| `nts.agents` | agent configs: `READER` (read-only questions), `COORDINATOR` (questions and changes) | done |
| `nts.eval` | scenarios = start state + request + checks on the committed state; paired runs vs the old pipeline | later |
| `nts.api` | FastAPI + server-sent events of each turn | later |

## Tools, by scope

- **read**: `find`, `describe`, `classes_of`, `free_hours`, `who_is_free`, `free_rooms`, `changes` (done)
- **draft**: `constrain` (unavailable / no_classes / classes_only / room_needed), `move_class`, `retract`, `list_draft` (done)
- **evaluate**: `check_draft` -> a repair solve of every timetable the draft touches: ok or not, the moves, who else is
  affected, wishes not met; when it clashes, the clashing rules in words, marked as the draft's or already agreed (done)
- **talk**: `ask_requester`, `message_party`, `make_offer` (the episode waits for the answer)
- **end**: `submit` (only a draft checked at exactly this version), `decline`, `escalate` (done)

Who may change what: a teacher only their own times, unshared courses and classes; the office (`coordinator`)
anything, including sections and rooms. Tiers: the office's firm rules 2, firm with a reason 3, firm 4, wishes 5.
The commit gate checks every timetable a new rule touches, not just the ones the draft wrote.

Tools take and return the department's names ("Dr. Iyer", "Lab 2", "Tue", "2 pm"), never ids. A name that
matches nothing is refused with `did_you_mean`, so the model fixes the call on its next turn.

## Build order

1. Kernel, state store, read tools, scripted model, tests. **(done)**
2. Harness loop + budgets, telemetry, replay model: every test deterministic. **(done)**
3. Draft and evaluate tools; the coordinator end to end on Gemini (clean first traces). **(done)**
4. Exporters; a 4B LoRA on the traces; the same scenarios on the local model.
5. Talk tools, negotiation with simulators, API and UI.

## Running it

    python -m nts init --store runs/agentic/demo.jsonl          # a generated department, solved
    python -m nts ask "I have a conference next week on Tuesday and Wednesday" --as "Dr. Iyer" --week 5         --env ../negotiate-then-solve/.env --show --approve
    python -m nts ask "..." --model llama --url http://10.1.25.191:8080/v1 --adapter <lora name>

(`PYTHONPATH=src` and the main repo's venv when uv cannot install offline.) A recorded episode replays on the
store as it was: `store.fork(span.attrs["state_version"])` plus `ReplayModel.from_spans(...)`.

## Running the tests

    uv run pytest tests/nts        # or, offline: ../negotiate-then-solve/.venv/Scripts/python -m pytest tests/nts

## Telemetry

Every episode is recorded as spans under `runs/telemetry/` (gitignored: the records hold real names):
`spans/YYYY-MM-DD.jsonl` is the record itself, append-only; `index.sqlite` holds the same spans to query.

| Span | Holds |
|---|---|
| episode | agent and its version hash, model, who asked, week, request, state version, tools offered, budgets; status, answer, counts, tokens, time |
| turn | its number, whether it was the last, the tools offered |
| model_call | the exact messages sent, the request hash, the reply as the model gave it, tokens |
| tool_call | the call, the state version it ran on, the result or the error |

`ReplayModel.from_spans(load_trace(folder, episode_id))` reruns an episode with no server; in strict mode it stops
with `ReplayDiverged` at the first model call that is asked differently from the recorded one.
