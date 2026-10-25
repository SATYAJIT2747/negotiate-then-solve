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
| `nts.tools` | typed tools (pydantic args -> JSON schema), scopes, `Refusal` errors with hints, name resolution | done |
| `nts.people` | who answers the talk tools: simulators (`Cast` of a `Requester` and `Teacher`s), `Terminal` | done |
| `nts.protocol` | messages, tool calls, results: what every layer passes and telemetry records | done |
| `nts.models` | one `chat(messages, tools)` interface: scripted (tests), replay, llama.cpp (`OpenAIChat`, LoRA per request), Gemini | done |
| `nts.harness` | the loop: budgets (turns, calls, tokens, time), repeat refusal, context trimming, waiting for answers (pause/resume) | done |
| `nts.telemetry` | spans episode > turn > model_call / tool_call; JSONL per day + SQLite index | done |
| `nts.data` | episodes rebuilt from telemetry, labels (scenario verdict > approval > none), SFT / DPO exporters | done |
| `nts.agents` | agent configs: `READER` (read-only questions), `COORDINATOR` (questions, changes, talk), `COORDINATOR_V1` (no talk) | done |
| `nts.eval` | scenario families from the department's own names; checks on the committed state; runner + report | done |
| `nts.api` | FastAPI: episodes in threads, their spans as server-sent events, inbox and replies, approval; one web page | done |

## Tools, by scope

- **read**: `find`, `describe`, `classes_of`, `free_hours`, `who_is_free`, `free_rooms`, `changes` (done)
- **draft**: `constrain` (unavailable / no_classes / classes_only / room_needed), `move_class`, `retract`, `list_draft` (done)
- **evaluate**: `check_draft` -> a repair solve of every timetable the draft touches: ok or not, the moves, who else is
  affected, wishes not met; when it clashes, the clashing rules in words, marked as the draft's or already agreed (done)
- **talk**: `ask_requester` (a question to the person asking), `make_offer` (ask the owner of a rule that blocks
  the draft to waive it for the weeks it blocks), `message_party` (tell someone; no answer awaited) (done)
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
4. Exporters; a 4B LoRA on the traces; the same scenarios on the local model. **(done: `nts-agent-v1-read-100` passes 17/18 on `eval.jsonl` seed 0; base 4B 12/18, Gemini 3.5 Flash-Lite 15/18)**
5. Talk tools, negotiation with simulators, API and UI.
   - 5a. Talk tools, waiting for answers, simulated people, the `clarify` / `negotiate` / `negotiate_refused`
     families. **(done)**
   - 5b. API: FastAPI, server-sent events of each turn, a person answering an episode that waits. **(done)**
   - 5c. UI on the API: `nts/api/static/index.html`, served at `/`. **(done)**

## Talking to people

A talk tool returns an `Ask`; the harness hands it to `people` (`harness.run(..., people=Cast(...))`) and the
model sees the `Reply` as the tool's result. With no people the episode stops as `waiting` (`outcome.pending`
holds the ask) and `harness.resume(outcome, Reply(...))` goes on from there, so a person can answer later. Calls
made in the same turn after an ask are refused (`made_before_answer`): they were made before the answer.

Negotiation is one move, checked by the kernel like everything else. When `check_draft` finds the draft
clashes with a rule another teacher set, it lists that rule with its owner and a number; `make_offer(to, rule,
message)` asks the owner to waive it for the weeks it blocks. A yes writes the waiver into the draft (the rule
again, with those weeks in `valid.except_weeks`), the agent checks the draft again and submits. A no is final for
that rule (`already_refused`). Rules the office set ("already agreed") are not offered: those are escalated. A
teacher's draft cannot otherwise touch another teacher's rules.

## Running it

    python -m nts init --store runs/agentic/demo.jsonl          # a generated department, solved
    python -m nts ask "I have a conference next week on Tuesday and Wednesday" --as "Dr. Iyer" --week 5         --env ../negotiate-then-solve/.env --show --approve
    python -m nts ask "Please move my class to Friday" --as "Dr. Iyer" --week 5 --talk     # you answer its questions
    python -m nts ask "..." --model llama --url http://10.1.25.191:8080/v1 --adapter <lora name>

(`PYTHONPATH=src` and the main repo's venv when uv cannot install offline.) A recorded episode replays on the
store as it was: `store.fork(span.attrs["state_version"])` plus `ReplayModel.from_spans(...)`.

## The API and the page

    python -m nts serve --store runs/agentic/demo.jsonl --model llama --adapter <lora> --port 8100
    python -m nts serve --model gemini --env ../negotiate-then-solve/.env --simulate   # stand-ins answer asks

Open http://127.0.0.1:8100/. Pick who you are (a teacher or the timetable office), send a request and watch
the agent's turns arrive. When it asks something, the episode waits in the inbox of the person asked (the
requester, or the teacher whose rule it wants waived), who answers on the page; the office approves a
submitted draft, which commits it through the kernel. Each episode runs in a thread; every span the harness
writes becomes an event (`model`, `tool`, `reply`, `waiting`, `end`, `decision`) on
`GET /api/episodes/{id}/events` (server-sent events; the stream ends when the episode ends or waits, and the
client reconnects with `?after=n` once it has replied). `GET /api/episodes/{id}/poll` gives the same as JSON.
Who is calling is taken on trust (`actor`, `person`); behind a real login the server would take it from the
session. Approving commits to the store file, so try things on a copy of a store you want to keep.

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

## Scenarios, labels and training data

`python -m nts eval --model gemini --per-family 3 --seed 0` builds a suite from the store's department (9
families: two kinds of question, away for a day, a soft preference, moving one class, a room closed by the
office, a request about someone else, an impossible request, an ambiguous one; then three with talk: an unclear
request the requester explains when asked, a move blocked by another teacher's firm rule that they agree to waive,
and one they refuse), runs each on a fork of the store (plus the scenario's setup rules, with its simulated
people), approves what is submitted, judges the committed state, and records an "approval" and an
"evaluation" span on every episode. `--agent coordinator-v1` runs the nine families without talk, the suite
`nts-agent-v1` was evaluated on (the first nine families build the same scenarios as before). The report goes to `runs/eval/<model>-s<seed>.json`.

An episode's label is the strongest judgement on its trace: the scenario verdict, else an approval or
rejection (`harness.approve` / `harness.reject`, which the CLI's `--approve` uses), else none.
`python -m nts export --out data/agent-v1 --models gemini` writes:

- `sft-train.jsonl`, `sft-val.jsonl`: good episodes as `{"messages", "tools", "meta"}` (split by a hash of
  the request); turns whose calls were all refused for a fixable reason are left out (`--keep-errors` keeps them)
- `dpo.jsonl`: refused turn vs the turn that fixed it, with the conversation before as the prompt

`notebooks/train_agent_kaggle.ipynb` trains the adapter on that: one example per assistant turn, rendered with
the tools by Qwen's chat template, loss on the turn only.

Prompt size: the tool schemas are compacted when generated (no titles, no `anyOf [X, null]`, no empty
defaults): 2,201 tokens for the coordinator's 15 tools, down from 2,885. The context budget counts them. A
local model gets no token budget (`--max-tokens` sets one).
