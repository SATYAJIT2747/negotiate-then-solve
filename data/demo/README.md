# Demo departments for the web app

Four synthetic departments, each a store (the append-only event log `nts serve` reads and writes):

| file | department | what is in it |
|---|---|---|
| `cse-s1.jsonl` | cse-s1 | 8 teachers; one change already committed (Dr. Iyer is away at one hour, week 6) |
| `cse-s11.jsonl`, `cse-s12.jsonl`, `cse-s13.jsonl` | cse-s11..13 | 8 teachers each, the timetable as generated |

The server writes to the store it serves, so copy them first and serve the copies. To start over, copy again.

```sh
git clone -b agentic https://github.com/Seronic2001/negotiate-then-solve.git
cd negotiate-then-solve
uv sync                                   # or: python -m venv .venv && pip install -e .
mkdir -p runs/agentic && cp data/demo/*.jsonl runs/agentic/
```

## Run it

With the local llama.cpp server (the base model, with the agent adapter in its `--lora` list):

```sh
uv run python -m nts serve --store runs/agentic/cse-s1.jsonl runs/agentic/cse-s11.jsonl \
    --model llama --url http://127.0.0.1:8080/v1 --adapter nts-agent-v2-read-100 --agent coordinator-v2
```

- `--adapter` is the adapter's file name without `.gguf`. With only `nts-agent-v1-read-100` loaded, use
  `--agent coordinator-v1` (no asking or offers: it drafts changes and sends them to the office).
- With Gemini instead: `--model gemini --env .env` (a `.env` with `GEMINI_API_KEY=...`); the default agent
  (`coordinator`, v3) then also answers questions about the rules and the fairness ledger.
- `--simulate`: stand-ins answer the agent's questions and offers, so nobody has to wait for a reply.

Open http://127.0.0.1:8100/. Sign in by department and name, with no password:

- a teacher (for example "Dr. Iyer"): make requests, answer the agent's questions;
- `office`: approve or reject drafts, Bookings, Semester rebuild, Observability;
- `club: Robotics` (any club name): book rooms.

## On a network

Without `--users`, anyone can sign in as anyone, which is fine only on one machine. To serve others, set
passwords and listen on all interfaces:

```sh
uv run python -m nts passwd office --dept cse-s1            # repeat for each person who signs in
uv run python -m nts serve ... --users runs/agentic/users.json --host 0.0.0.0
```
