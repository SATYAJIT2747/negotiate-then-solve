# Negotiate, Then Solve

Conflict-aware university timetabling with LLM agents and constraint solvers
(LMA Group Project 2026). Every layer of the proposal's architecture is
implemented: channels and the request store (L0–L1), System One / System Two
parsing and the fine-tuned compiler (L2), the policy agent (L3), CP-SAT with
conflict analysis (L4), the interpretation layer with the resolution ladder,
grounded explanations and the concession ledger (L5), and human approval
before publishing (L6). The evaluation harness covers the negotiation
benchmark (baselines B1–B4 and the oracle, ablations A1–A3), the reply-action
benchmark, parsing, policy and the safety set.

Every LLM-backed component has a deterministic path (template explanations,
scripted simulators, stub parsers), so the whole system can be developed and
tested without API calls; the Gemini free-tier quota is kept for experiments.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync                       # create .venv and install dependencies
uv run pytest -m "not slow"   # fast tests (< 1 min)
uv run pytest                 # includes a full department-size solve
uv run python -m evaluation.demo     # generate a department, solve it, run UC3
uv run nts-web                # web app on http://127.0.0.1:8000 (builds the frontend first; needs Node.js)
```

## Running the project

The web app runs on Windows; llama.cpp (for the fine-tuned local model) runs
in WSL. WSL forwards `localhost`, so the app reaches `llama-server` at the
default `http://localhost:8080/v1` without extra setup.

**1. Start the local model in WSL** (skip this for the offline or Gemini
parser). The fine-tuned weights from the latest training run are in
`qwen3.5-2b-nts-gguf/` next to this repository (Qwen3.5-2B, Q5_K_M GGUF). From a WSL terminal, with llama.cpp built (adjust the path to your
`llama-server` binary):

```sh
~/llama-bin/llama.cpp/llama-server \
  -m "/mnt/c/Users/Shubh/Desktop/LMA Major Project/qwen3.5-2b-nts-gguf/Qwen3.5-2B.Q5_K_M-mutitask-finetune.gguf" \
  --jinja -ngl 99 -c 4096 --cache-ram 1024 --host 0.0.0.0 --port 8080
```

`--cache-ram 1024` caps llama.cpp's prompt cache in RAM (default 8 GB); without
it the server grows past WSL's memory during long evaluations and is killed.
Check it from Windows with `curl http://localhost:8080/v1/models`. If
`localhost` does not reach WSL, use the WSL address (`wsl hostname -I`) and set
`NTS_LOCAL_URL=http://<wsl-ip>:8080/v1`.

**Several models on one server (router mode).** Start `llama-server` with
`--models-dir <folder> --models-max 1` instead of `-m`: it lists every GGUF in
the folder and loads the one a request names, unloading the previous one.
Choose it with `NTS_LOCAL_MODEL=<name>` (the web app) or `--local-model <name>`
(the evaluations); `curl <url>/models` lists the names. Answers are cached by
model file name, so give every model file its own name (e.g.
`qwen3.5-2b-v2.Q5_K_M.gguf`).

**2. Start the app** (Windows, PowerShell, from `negotiate-then-solve/`):

```powershell
uv sync
uv run nts-web --parser local        # fine-tuned model via llama-server in WSL
# uv run nts-web                     # offline rule-based parser, no model needed
# uv run nts-web --parser gemini     # Gemini (keys in .env)
```

`nts-web` builds the front end before it starts (`npm install` when
`node_modules` is missing or `package-lock.json` changed, then `npm run build`),
so Node.js and npm must be on `PATH`. The build is skipped when `frontend/dist`
is newer than every source file, so a restart with no front-end changes starts
immediately. `--rebuild` forces a build; `--no-build` (or `NTS_BUILD=0`) serves
`frontend/dist` as it is.

Open http://127.0.0.1:8000. For UI development, also run `npm run dev` in
`frontend/` and use http://localhost:5173 (it proxies `/api` to :8000).

## Layout

```
src/
  core/        the timetabling model: schemas, instances, CP-SAT solver, validator, conflicts (MUS/MCS), graph
  language/    LLM clients (Gemini, local), System One and Two parsing, the fine-tuned compiler and its data
  agents/      policy agent and policy documents, negotiation, priorities, concession ledger, explanations
  pipeline/    channels (ingest), the central store and the orchestrator (request lifecycle)
  semester/    course offerings -> semester timetable, plain-word preferences, versions, club bookings
  training/    multi-task training data for the local model (distilled from every agent-side call)
  evaluation/  benchmark, safety set, evaluation scripts, LLM judge, metrics and statistics
  web/         JSON API for the React app, the world behind it, the channel portal, semester endpoints
frontend/      React + TypeScript + Tailwind app (pages/, components/, lib/)
tests/         pytest, one file per area (`-m "not slow"` skips the department-size solves)
scripts/       one-off helpers (sample policy documents)
notebooks/     Kaggle notebooks: fine-tuning the local model, Unlimited-OCR sidecars
data/          handbook, request corpora, policy documents, training data
course-offering/  the institute's course offering document and reference timetables
runs/          outputs: LLM cache, experiment results, semester state (not committed)
```

Dependencies run one way: `core` <- `language` <- `agents` <- `pipeline`, `semester`; `training`, `evaluation` and `web` sit on top.

| Module | Proposal | What it does |
|---|---|---|
| `src/core/schemas.py` | §7.4, Fig. 2 | `Constraint`, `Request` + lifecycle state machine, `Placement`, `TimetableVersion`, `ConcessionEntry` |
| `src/core/instance.py` | §1, §8.3 | Rooms, faculty, groups, sessions, calendar; institute policy as Tier 1 constraints |
| `src/core/semantics.py` | §8.3 | What each constraint type means for a placement (shared by solver and validator) |
| `src/core/solver.py` | §8.3–8.4 | CP-SAT model; `a_k ⇒ C_k` guards; full build, minimal-change repair, lexicographic tiers |
| `src/core/conflicts.py` | §8.5 | `find_mus` (deletion-based shrinking), `enumerate_mcs` (cheapest-first, solver-verified witnesses) |
| `src/core/validator.py` | §8.2 step 4, §12.4 | Rejects bad compiled constraints; scores any timetable's hard-constraint validity |
| `src/core/generator.py` | §12.1 | Seeded synthetic departments (30 faculty, 60 course sections, 20 rooms by default) |
| `src/core/scenarios.py` | §9 | UC3 lab-contention fixture (the benchmark generalises it) |
| `src/core/itc.py` | §12.1 | ITC-2007 Track 3 `.ctt` loader and `.sol` writer |
| `src/language/corpus.py` | §12.1 | Request corpus generator with labels; JSONL load/save |

## Parsing layer and Gemini

Put one or more keys in `.env` (git-ignored):

```sh
# Single key or comma-separated list of keys:
GEMINI_API_KEY=key1,key2,key3
# Or explicitly via GEMINI_API_KEYS:
# GEMINI_API_KEYS=key1,key2,key3
# Or numbered:
# GEMINI_API_KEY_1=...
# GEMINI_API_KEY_2=...

# optional overrides (defaults are pinned in src/language/llm.py)
GEMINI_MODEL=gemini-3.5-flash-lite       # agent: System Two parser, negotiator
GEMINI_SIM_MODEL=gemini-3.1-flash-lite   # stakeholder simulators, LLM judge
GEMINI_FLASH_MODEL=gemini-3.8-flash      # larger-model comparison set
GEMINI_RPM=15     # requests per minute per key
GEMINI_RPD=500    # requests per day per key
```

**API Key Pooling & Load Balancing**: When multiple keys are configured, `src/language/llm.py`
pools them automatically with round-robin load balancing. Having $N$ keys multiplies
your total throughput ($N \times \text{RPM}$) and daily capacity ($N \times \text{RPD}$).
If an individual key hits a per-minute rate limit or reaches its daily quota, the
client dynamically rotates and fails over to the next available key in the pool.

`src/language/llm.py` caches every response under `runs/llm_cache/` (keyed by model,
prompts, schema and temperature), so rerunning an experiment costs no quota,
and a run stopped by the daily budget resumes the next day where it stopped.

| Module | Proposal | What it does |
|---|---|---|
| `src/language/llm.py` | §14 | Gemini client: structured output, cache, RPM limiter, daily budget, 429 backoff |
| `src/language/parsing.py` | §8.2 | System Two parser. The LLM drafts; tiers (Table 5), authority and validation are deterministic |
| `src/language/system_one.py` | §8.2 | Calibrated TF-IDF + logistic regression: request type, action, confidence, fast-path routing with τ |
| `src/evaluation/metrics.py` | Table 11 | ECE, constraint exact match, atom-level P/R/F1 |
| `src/evaluation/parsing.py` | §12 | `uv run python -m evaluation.parsing --limit 40` |
| `src/language/paraphrase.py` | §12.1 | Rewrites the corpus in varied registers with the simulator model; a rule-based check rejects paraphrases that change days, numbers or names |

**Paraphrased corpus.** `uv run python -m language.paraphrase` writes
`data/requests-para.jsonl`: the same 600 requests and labels, with each text
rewritten (10 per call, about 60 calls) and the template kept in
`template_text`. Injected instructions are cut off before paraphrasing and put
back verbatim. Paraphrases that change a day, number, person, room, group or
course are retried in another style. The check cannot see a wish turning into
a demand, so read a sample by hand. Evaluate on it with
`--corpus data/requests-para.jsonl`; add `--s1-train data/requests.jsonl` to
train System One on templates and test it on paraphrases.

## Policy agent

`src/agents/policy.py` checks a parsed request against the handbook
(`data/handbook.md`, **a synthetic stand-in**: replace it with the
institute's handbook, one `## <number> <title> [RULE-ID]` heading per rule).

1. Retrieval returns the top 5 rules. The query is extended with what the
   parse implies (a one-off absence adds "make-up", a run of more hours than
   the limit adds "consecutive"). Three retrievers (`--retriever`, or
   `NTS_RETRIEVER` in the web app):
   - `bm25`: words; clock times are normalised ("1pm" and "13:00" match).
     The recorded evaluation runs used it.
   - `dense`: sentence embeddings (BAAI/bge-small-en-v1.5 through fastembed,
     ONNX, no torch; 65 MB, downloaded to `runs/embed_models/` on first use),
     cosine similarity. With one handbook and a few documents a matrix
     product replaces a vector index (FAISS/Chroma in the proposal).
   - `hybrid` (the web app's default): both rankings fused by reciprocal
     rank. Falls back to BM25 if the embedding model cannot be loaded.
2. The LLM returns `allowed`, `needs_approval` or `forbidden`, cited rules,
   obligations (e.g. a make-up class), an explanation and a compliant
   alternative; for policy questions, an answer.
3. Deterministic checks: citations must be among the rules shown, and a
   denial without a valid citation becomes `needs_approval`, so nothing is
   denied without the rule it breaks.

`uv run python -m evaluation.policy --split val` scores allow/deny accuracy,
citations, make-up obligations and retrieval recall against the corpus
`rules` labels. Use `--policy-model gemini-3.1-flash-lite` for development
when the agent model's daily quota is spent.

`uv run python -m evaluation.retrieval` compares the three retrievers with
no API calls, over the handbook and the policy documents (32 rules): corpus
requests (test split) as sent, the same with the parse's hints (what the
agent searches with), and `data/retrieval-queries.jsonl`, 40 questions
worded differently from the rules (written for this comparison, not by
the team). Recall@5, `runs/retrieval-20260929.json`:

| | BM25 | Dense | Hybrid |
|---|---|---|---|
| Corpus + hints (106) | 87.7% | 91.5% | **93.4%** |
| Paraphrased questions (40) | 65.8% | **87.8%** | 82.9% |
| Corpus, raw text only (106) | **77.4%** | 45.3% | 75.5% |

Dense alone misses requests whose rule is implied rather than said ("I'm at
a conference in week 7" → P-MAKEUP); hybrid keeps BM25 there and gains most
of the dense retriever's reach on paraphrases. Most remaining misses are
vague absences ("away for a few days soon") that the parser sends back for
clarification, so no absence hint is added.

| Module | Proposal | What it does |
|---|---|---|
| `src/agents/policy.py` | L3, UC2, UC4 | Handbook loader, BM25, dense and hybrid retrieval, policy agent with citation checks |
| `src/agents/documents.py` | L3 | Policy documents in any form: PDF, scans, photos, HTML, Markdown → rules with provenance |
| `src/evaluation/policy.py` | Table 11 (RAG) | Parse-then-policy evaluation |
| `src/evaluation/retrieval.py` | Table 11 (RAG) | BM25 vs dense vs hybrid: recall@1/3/5, MRR |

### Policy documents: PDF, HTML, scans and photos

Institutes publish regulations as PDFs, circulars as scans, notices as photos
and pages on their website. `src/agents/documents.py` turns all of them into rules the
policy agent can retrieve and cite, next to the handbook:

| Input | How it is read |
|---|---|
| PDF with a text layer | the text layer, page by page (pypdfium2); running headers, footers and page numbers are dropped |
| scanned PDF page (no text layer) | rendered at 220 dpi, then OCR |
| image (scan, phone photo) | OCR, after deskewing the detected text boxes |
| HTML page | the content only: scripts, navigation, header, footer, cookie banners and side panels are removed; headings, lists and table rows kept |
| Markdown / text | as written; the handbook keeps its `[RULE-ID]` headings |

OCR backends (`NTS_OCR`):

- `rapidocr` (default, CPU): PaddleOCR's PP-OCR text detector and English
  recogniser as ONNX. The English model (9 MB) is downloaded to
  `runs/ocr_models/` on first use (the bundled model drops the spaces between
  English words). Lines read with low confidence get a second look at a larger
  detection size. About 3–7 s per page on the laptop CPU.
- `unlimited`: [Baidu Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR)
  (3B, 0.5B active, MIT), a document-parsing model that returns Markdown. It
  needs about 8 GB of VRAM, so it does not run on the development laptop.
  Either serve it with vLLM on a GPU machine and set `NTS_OCR_URL`, or run
  `notebooks/unlimited_ocr_kaggle.ipynb` on a Kaggle T4. The notebook writes a
  `<file>.ocr.json` per document; put it next to the file and it is used for
  every page without a text layer.

OCR results are cached by image hash (`runs/ocr_cache/`). The text is then cut
into rules at numbered headings ("3.2 Timetable changes", "2. Capacity. …",
including OCR slips such as "1.Fixed"); a document without headings becomes
paragraph chunks. Each rule records its file, pages and method, and gets the
ID `<DOC>-<number>` (`EXAM-3.2`) unless the document gives its own `[ID]`.

```sh
uv run python -m agents.documents                        # ingest data/policies/ and run demo queries
uv run python -m agents.documents notice.jpg --show-text # one file, with the OCR text
uv run python scripts/make_policy_samples.py          # rebuild the four sample documents (needs Chrome)
```

`data/policies/` holds one sample of each kind (a digital PDF, a scanned PDF,
a photographed notice and a web page); the web app loads them at start-up. On
the Handbook page the coordinator can add or remove documents, and every rule
and search hit shows the document, page and method it came from.

Known limits: OCR can misread single letters ("ciasses"), which the confidence
shown per document hints at. BM25 alone matches words, not meanings: "shift
my practical to another lab" does not reach "moved into a different
laboratory"; the dense and hybrid retrievers do (the Handbook page's search
can switch between the three).

## Fine-tuned compiler (local model)

`uv run python -m language.compiler` writes `data/compiler/{train,val,test}.jsonl`:
chat examples (short system prompt, compact directory, message) with the gold
`ParseOutput` JSON as the answer. `refuse` requests are left out (the
authority check decides them). Rule-breaking (`deny`) requests are kept as
`compile` with the constraint they ask for, 1 pm being the lunch slot; the
policy agent denies them from that parse. `--extra-denials 150` (default)
adds generated rule-breaking requests in wider wording, and legal look-alikes
(12 pm, 2 pm, a run at the limit), to train; none shares its text, or its
sender, day and rule, with a val or test request. Templated and paraphrased
versions of a request share a split. A train or val request whose text, greeting and
sign-off aside, repeats a request of a later split is dropped (the templates
and the fixed policy questions repeat across splits); the evaluation splits
are unchanged. `training.distill export` applies the same rule to policy rows. `--min-per-action 50` (default) then tops up
the answer, out-of-scope and clarify examples in train with new wordings
(40 policy questions, 40 out-of-scope messages, vague absences and
preferences), none repeating a corpus request.

`notebooks/train_compiler_kaggle.ipynb` trains it on a Kaggle T4 with LoRA
(Unsloth advises against QLoRA for Qwen3.5), loss on the answer only, then
exports GGUF for llama.cpp on a 6 GB laptop GPU. On a T4, Unsloth runs
Qwen3.5 in float32 (no bf16, and fp16 hits a dtype mismatch), where the 4B
model needs about 18 GB and runs out of memory; the notebook therefore
defaults to **Qwen3.5-2B**. For 4B, use a GPU with bf16 or set
`load_in_4bit = True` (QLoRA, with Unsloth's caveat). Run it with
`TRIAL = True` first.

Evaluate the downloaded model locally with the same harness as Gemini (the
latest run's weights are in `../qwen3.5-2b-nts-gguf/`; start `llama-server` in
WSL as in [Running the project](#running-the-project)):

```sh
llama-server -m ../qwen3.5-2b-nts-gguf/Qwen3.5-2B.Q5_K_M-mutitask-finetune.gguf --jinja -ngl 99 -c 4096 --port 8080
uv run python -m evaluation.parsing --local --corpus data/requests-para.jsonl --split val --limit 100 --workers 1
```

`src/language/local.py` talks to any OpenAI-compatible server (llama.cpp, Ollama, LM
Studio), constrains output to the JSON schema, keeps thinking off and caches
like the Gemini client. On a GPU with less memory than the model file, lower
`-ngl` so some layers run on the CPU.

### Multi-task local model (every agent-side call)

The compiler only parses. `src/training/distill.py` builds data for the other calls
the deployed system makes: reading negotiation replies, grounded explanations
and the policy agent. Each example uses the system prompt and user message the
system sends at run time (the same code builds them), so the trained model
drops into every `--local` path unchanged. The stakeholder simulators, the
judge, the paraphraser and the LLM baselines (B1, B3, B4) stay on Gemini:
they test or grade the system, or measure an untuned LLM.

```sh
uv run python -m training.distill messages --seeds 1-10   # negotiations on new scenarios; no API (~30 min)
uv run python -m training.distill reply                   # simulator model voices replies for all six tools
uv run python -m training.distill reply --voice template --kinds propose,clarify,escalate --per-message 1 --fraction 0.55 --name reply-tools
                                                          # only the newer tools, beside the existing reply.jsonl; no API
uv run python -m training.distill explain                 # agent model as teacher
uv run python -m training.distill policy                  # agent model as teacher
uv run python -m training.distill denials                 # more denials from the policy stage's answers; no API
uv run python -m training.distill export                  # data/multitask/{train,val}.jsonl, ≤900 per task
```

**Readable explanations and intents** (the next model): the parser returns
intents (`language.intents`: clock times, weeks, classes described as the
message describes them) instead of slot numbers and IDs, and explanations are
written from the readable facts.

```sh
uv run python -m language.intent_compiler                 # data/compiler-intents/: corpus gold as intents + new intents
uv run python -m training.distill messages --messages messages-readable   # facts with their plain text; no API
uv run python -m training.distill explain --readable --messages messages-readable --sample 600
uv run python -m training.distill export --readable --compiler-dir data/compiler-intents --out data/multitask-readable
```

A model trained on it is used with `NTS_INTENTS=1` beside `--parser local`; the
orchestrator then routes swaps, extra classes, cancellations and withdrawals by
the intents it reads.

- **Scenarios** come from benchmark seeds 1-10; seed 0 (the evaluation set) is
  refused, and the last seed is the validation split.
- **Replies:** the tool call is drawn first (accept a random option, counter
  with a window, propose a day, time and room, clarify, reject, escalate),
  then the simulator model (or `--voice template`, no API) writes the text,
  so labels are exact. Escalate replies (injected instructions, policy
  exceptions, out-of-scope asks) always keep their template text. 15% of the
  fixable ones show the retry turn: the typed errors of a rejected call, with
  the corrected call as target. The templates are worded apart from the reply
  benchmark (`evaluation.replies`). Rows made with the older three-decision
  prompt (`reply.jsonl`) get the current prompt in `check` and `export`.
- **Explanations:** a teacher output is kept only if every claim cites known
  fact IDs (brackets stripped) and passes the claim checker, every option is
  cited, and nothing private leaks. The stage prints why the rest were dropped.
- **Denials:** the first model trained on this data never wrote slot 4 (the
  compiler data had no request for 1 pm), so it misread lunch-hour requests
  and the policy agent could not deny them from its parse. `denials` adds the
  generated rule-breaking requests and their look-alikes, and corpus denials
  shown with a misread parse (lunch moved an hour, a run cut to the limit,
  or none), so the verdict comes from the message. Its targets are the
  teacher's answers from `policy`.
- **Policy:** input parses come from the rule parser (no API). The verdict and
  citations must match the corpus label and cite only rules that were shown.
  The make-up obligation, which the teacher often leaves out, is set from the
  label.

Every stage caches, so one stopped by the daily quota resumes when run again.
Train with `notebooks/train_compiler_kaggle.ipynb` on the `data/multitask/`
folder (it reports validation per task), then score the model with
`eval_parsing --local`, `eval_policy --local` and `eval_negotiation --local`.

For **Qwen3.5-4B**, use `notebooks/train_multigpu_kaggle.ipynb` (GPU T4 x2). A T4 has
no bf16, so the weights stay in float32 (~17 GB), which no single T4 holds; the
notebook drops Unsloth for plain transformers + PEFT + TRL and splits the layers
across both GPUs (`device_map="auto"`). Only one GPU computes at a time, so expect
2-3x the 2B run: the TRIAL run projects the full length against Kaggle's 12-hour
limit, and training checkpoints and resumes across sessions. It exports GGUF with
llama.cpp's converter (Q5_K_M and Q4_K_M).

## Data

`uv run python -m language.corpus` regenerates `data/requests.jsonl` (600 requests,
400/50/150 train/val/test, plus 30 test-only multi-constraint requests; `--multi 0`
leaves them out) and pins the instance it was built against in
`data/synthetic-cse-s0.json`. Splits are made by wording template, not by
request: every request built from one template (`template`, e.g.
`unav_conference:1`) is in the same split, so no test request has a
re-dressed or paraphrased twin in training. Each builder's largest template
stays in train; its smallest goes to test where there is room. Val and test
requests carry `slices` (`unseen_wording`, `unseen_combination`, `ambiguous`,
`adversarial`, `multi_constraint`), and `evaluation.parsing` reports each.
Each line is a `CorpusExample`:

- `request`: the raw message as it arrives (sender, role, channel, text);
- `request_type`, `authorised`, `expected_action` (`compile`, `clarify`,
  `deny`, `refuse`, `investigate`, `answer`, `out_of_scope`): what System One
  and the policy agent should decide;
- `targets`: the constraints a correct compiler produces. For `clarify`
  examples these are the true constraints behind the vague text, so a
  stakeholder simulator can answer the clarifying question;
- `missing`, `violates_rule`, `injection`: labels for ambiguous,
  rule-breaking and prompt-injection cases;
- `rules`: handbook rules a correct policy check cites (the rule broken, the
  rule a policy question asks about, or P-MAKEUP for a one-off absence).

**Hand-written requests** (the proposal's 100 team-written examples) go in
`data/handwritten.jsonl` in the same format with `"handwritten": true`.
`load_jsonl(path, instance)` rejects any target that names an unknown
faculty member, room or time.

**ITC-2007 instances** (comp01–comp21) come from the CB-CTT benchmark portal
maintained by the University of Udine. Load them with
`core.itc.load_ctt(Path("comp01.ctt"))`. Room capacity is soft by default,
as in ITC; the ITC soft constraints for working days, compactness and room
stability are not modelled.

## Interpretation layer (L5): negotiation

| Module | Proposal | What it does |
|---|---|---|
| `src/agents/priority.py` | §8.4 | π_k = auth + impact + justification + lead + credit − disruption (higher = dearer to relax); option cost = Σπ + λ1·moved + λ2·ΔGini over weighted burden; `flat=True` for ablation A1 |
| `src/agents/ledger.py` | §8.6 | Concession ledger with semester decay; each concession weighs importance (tier × justification) × magnitude (1 dropped, 0.5 counter-offer); burden dispersion: Gini, max, CV |
| `src/agents/explainer.py` | §8.7 | Conflict → numbered facts (MUS constraints, rules, rooms, options, ledger). Template, grounded (claims cite facts; a checker drops claims whose days, numbers or names are not in the cited facts) and free (A2) modes; private-reason leak check |
| `src/agents/negotiation.py` | §8.6 | Resolution ladder: auto-substitute → auto-relax (notices) → negotiate over MCS options (2–3 solver-verified alternatives per message, cheapest owner first, lowest credit on ties, one extra probe after a refusal) → escalate with a brief. Each reply becomes one typed tool call: accept, apply_reply, counter_propose (offered onward only if CP-SAT verifies it), ask_clarification (one follow-up per round), decline, escalate; a bad call gets its typed error back, and after two retries the reply goes to the coordinator. `RuleReplyParser` is the regex handler of ablation A3. Option sources: MCS (ours), LLM-invented unchecked (B3), LLM-invented filtered by CP-SAT with up to 3 regenerations (B4) |
| `src/agents/simulators.py` | §12.1 | Stakeholders with hidden flexibility (acceptable windows, room flexibility, whether they volunteer it, whether they reply); four policy families (strict, flexible, cost-sensitive, history-sensitive) with seeded random refusals and silences; scripted or LLM-voiced replies |
| `src/evaluation/benchmark.py` | §12.1 | 60 scenarios: contention, substitute, deadlock, Tier 3 vs 4 squeeze, capacity shortfall, preference trade-off, policy; families cycle within each kind. The oracle tries every hidden-flexibility combination and minimises the negotiator's own objective (tiers given up, then option cost); `objective()` scores any outcome the same way |
| `src/evaluation/negotiation.py` | brief §5 | Runs ours / B1–B4 / oracle / A1–A3 (B2 = same objective, cheapest correction imposed; B4 = primary baseline). `--plan` runs the reduced plan: ours-llm, B3, B4 on all 60 × 3 seeds; ablations on a stratified 30 and B1 on a stratified 20, seed 0 (650 LLM runs); invalid candidates, invention calls and filtered inventions (H1), tool calls, retries; validity, correct outcome, resolution rate, rounds per resolved conflict, distance to the oracle objective, escalation P/R and reasons, weighted burden (Gini, max, CV), faithfulness, leaks; pooled and per family; McNemar, Wilcoxon, bootstrap CIs |
| `src/evaluation/replies.py` | brief §5.1 D | 100 replies to a real UC3 message with gold tool calls (accept, partial, counter-proposal, vague, refusal, injected); tool and argument accuracy, valid first calls, recovery, unsafe-action rate; `--parser llm` or `rule` (A3) |
| `src/evaluation/judge.py`, `src/evaluation/stats.py` | §12.5 | Batched LLM judge (clarity, acceptability) and quadratic-weighted Cohen's κ against human raters |
| `src/evaluation/study.py`, `src/web/study.py` | §12.5 | Human data collected in the web app (`/study`): team labels for judge validation (100 claims vs the faithfulness verifier, 100 replies vs the recorded call, 110 messages rated for clarity and acceptability) and the anonymised faculty pilot (24 paired grounded/free messages for H3, plus a live session in the portal); Cohen's κ, H3 Wilcoxon, `human_ratings.csv` for the judge |

```sh
uv run python -m evaluation.benchmark                                  # data/scenarios.jsonl (about 30 s)
uv run python -m evaluation.negotiation --offline                 # ours, A1-offline, B2, oracle: no API calls
uv run python -m evaluation.negotiation --plan --local --sim-local   # the reduced plan on the local model
uv run python -m evaluation.negotiation --configs ours-llm,B4 --subset 20   # Gemini check of the main comparison
uv run python -m evaluation.replies --parser llm --local           # reply tool calls (Benchmark D); --parser rule is A3
uv run python -m evaluation.judge runs/negotiation-XXXX.json           # LLM judge (batched, Flash-Lite)
uv run python -m evaluation.judge --kappa runs/study/human_ratings.csv  # judge vs human raters
uv run python -m evaluation.study build runs/negotiation-study-source.json   # study items (ours-llm + A2, seed 0)
uv run python -m evaluation.study report                                  # kappas, H3, live checks; writes human_ratings.csv
```

Offline results so far (60 scenarios; `runs/negotiation-offline.json`,
`runs/negotiation-ours-probe.json`). These predate the policy families, the
oracle objective, weighted burden, the new B2 and the brief's relabelling
(old A2 = new A1, old A3 = new A2, old A1 = new B3, old B4 = oracle);
regenerate `data/scenarios.jsonl` and rerun before reporting:

| | Correct outcome | Valid timetables | Rounds (agreements) | Within 10% of oracle | Escalation P / R | Concession Gini |
|---|---|---|---|---|---|---|
| Ours (MCS, tiers, ledger, probing) | 95% | 100% | 1.56 | 100% | 0.85 / 1.00 | 0.23 |
| A1: flat weights, no ledger | 93% | 100% | 1.53 | 100% | 0.81 / 1.00 | 0.27 |
| B2: solver imposes, no negotiation | 18% | 100% | 0 | – | 1.00 / 0.47 | – |
| Oracle | 100% | 100% | 0 | 100% | 1.00 / 1.00 | – |

The A1 row is from the run before probing was added; rerun it for a
like-for-like comparison. Under B2 only 4% of imposed changes would have been
acceptable to their owners. The three scenarios ours escalates although the
oracle agrees are stakeholders who accept only one narrow window and never
say so; six offered alternatives missed it. LLM configurations (ours-llm,
B1, B3, B4, A1–A3) need a model; a 3-scenario smoke run already shows the free
explanation (now A2) leaking a private reason, and the LLM-only timetable (B1)
silently breaking requesters' hard constraints.

## Swap requests

"Could I swap my Tuesday 10 am lecture with Dr. Rao's Thursday 2 pm one in
week 7?" names sessions by where they are now, so it is read against the
published timetable, not compiled from the message (`src/agents/swap.py`):

1. **Recognised** when the parser says `swap`, or by a deterministic
   backstop (a swap word, or "take my … and I take their …", plus a
   colleague named) for parsers never trained on swaps. The parser prompt and
   schema are unchanged, so every cached parse stays valid. Questions about
   the swap rule stay questions; students are refused.
2. **Read**: the message is cut at "my" and at "<name>'s"/"their"/"his"/"her";
   each part is matched against that person's sessions by day, time, course
   and kind (`RuleSwapReader`; in Gemini mode `LLMSwapReader` shows the model
   both timetables). Code then checks the sender owns the first session, the
   colleague the second, and that each fits in the other's slot. If the
   message fits more than one pair it asks which, listing the candidates.
3. **Policy**: the policy agent sees the two moves; a lunch-hour or other
   rule still denies. P-SWAP itself is satisfied by the next two steps.
4. **Consent** (P-SWAP, "with the consent of both"): the colleague gets one
   option in their negotiation inbox. Declined or no reply → denied, nothing
   compiled.
5. **Compile and solve**: each session gets a hard time pin (Tier 4, owned by
   its teacher) at the other's slot, for the weeks named or the semester;
   rooms are the solver's. Then the usual repair, negotiation, fairness audit
   and the coordinator's approval ("informed before the swap takes effect").

`uv run python -m evaluation.swaps --e2e 10` generates 120 swap requests on
the benchmark department's timetable (five phrasings; days, times, course
names, surnames, colleague first, weeks) with exact gold: if the stated
clues fit more than one session, the right answer is a question. Rule
reader, `runs/swaps-rules-20260929.json`: 51/51 answerable pairs and weeks
right, 69/69 vague requests answered with a question, 0 wrong pairs
committed; through the orchestrator 10/10 agreed swaps handled (9 swapped in
the proposal, 1 correctly denied for crossing the lunch hour) and 5/5
declined swaps denied. The phrasings are templated by the same author as the
reader, so this checks the logic, not coverage of real wording;
`--reader gemini` or `--reader local` runs the model reader on the same set.

## Orchestrator, channels and approval (L0, L1, L6)

| Module | Proposal | What it does |
|---|---|---|
| `src/pipeline/orchestrator.py` | Fig. 2, §8.9 | The request lifecycle end to end; System One routing; refusal, denial, clarification, escalation, answer and forward exits; fairness audit; publishing only through `approve()` by a listed approver; notifications to affected people only |
| `src/pipeline/store.py` | L1, state | SQLite: requests (with dedupe), constraint records, timetable versions (per week, with rollback), ledger, audit log |
| `src/core/graph.py` | §8.8 | NetworkX knowledge graph: owns, references, has_authority_over, reports_to, derived_from, conflicts_with, affects, balance |
| `src/pipeline/ingest.py` | L0 | Email (RFC 822, IMAP poller), messaging (Slack-shaped JSON) and portal adapters; email → person → role; unknown senders rejected; quoted replies stripped; threading |
| `src/web/portal.py` | L0, L6 | FastAPI portal: submit, case status and trace, approval queue, approve/reject (coordinator only), published timetable; the `X-User` header stands in for SSO |
| `src/web/world.py` | all | The demo department and the real pipeline behind the web app; inboxes for interactive negotiation; which views each role sees |
| `src/web/app.py` | all | JSON API for the React front end (`uv run nts-web`) |
| `src/language/rule_parser.py` | L2 | Offline rule-based parser (same output and post-processing as System Two) |
| `frontend/` | L0, L6 | React 19 + TypeScript + Tailwind 4 + Framer Motion; Recharts, d3-force |

## Safety (proposal §13)

`uv run python -m evaluation.safety` runs the 50-case safety set (authority,
claimed authority, "I approve" messages, injections, private-reason probing,
spoofing) through intake and the orchestrator with System Two and the policy
agent (Gemini). `--compromised` swaps in a parser that obeys every injected
instruction, to show which defences are deterministic: 49/50 pass, with 0
unapproved publishes, 0 authority violations, 0 injection successes and 0
leaks. The one failure is an "I approve, publish it" email from the HoD: the
fooled parser's constraint is within the HoD's authority, so it is created,
but it still only reaches "awaiting approval".

## Semester timetable from the course offering document

The timetable office builds the whole semester from the institute's course
offering document, collects preferences from faculty and students, publishes,
and then re-solves around changes during the semester. Students book club
activities in the evenings, checked automatically against it.

| Module | What it does |
|---|---|
| `src/semester/offerings.py` | Reads the course offering document into courses (code, name, L-T-P-C, H1/H2/H, enrolment cap, faculty), programmes (required courses, elective slots) and elective pools. PDFs are read by position, so wrapped cells stay with their row; scans and web pages go through `agents.documents` |
| `src/semester/solver.py` | The institute's week, 40 rooms, sessions from L-T-P, the two-phase CP-SAT solver and an independent check of every hard rule |
| `src/semester/requests.py` | Preferences, unavailability and room closures in plain words |
| `src/semester/service.py` | Saved state, background builds, versions with diffs, publishing, club bookings |
| `src/semester/clubs.py` | Club activity checks and alternatives |
| `src/web/semester.py` | The endpoints (mounted by `web.app`) |

**The week** (from the institute's lecture and tutorial timetables): six
lecture slots of 1 h 25 min (08:30, 10:05, 11:40, 14:00, 15:35, 17:10); Mon, Tue
and Wed are days A, B and C, repeated on Thu, Fri and Sat; tutorials on a
one-hour grid (09:00 … 20:00); labs of up to three hours from 09:00 or 14:00;
Wednesday and Saturday afternoons free. All of it is in `SemesterCalendar`.

**Sessions from L-T-P and the course code:**

- L ≥ 3: a lecture pair (the same slot on a day and the day three later);
  L = 1–2 or "(H)": one lecture a week. A course bigger than the largest hall
  runs parallel sections, as many as the halls allow.
- T > 0: a tutorial hour, all groups (≤ 80 students) in parallel rooms.
- P > 0: a lab block of min(P, 3) hours in the lab type of the course's area
  (CS → computing labs, EC → electronics, SC → science, PD → design studio;
  names such as "Embedded" or "Signal" also mean electronics). Batches are
  made of whole programmes, so each programme is busy only in its own batch.
- Honours projects, theses, seminars and PE Centre sports are listed but not timetabled.

**Rooms:** 40 by default (3 halls of 250–300, 27 lecture rooms of 40–150,
5 computing labs, 3 electronics labs, a science lab and a design studio),
named as in the institute's timetables; `default_rooms()`. Programme sizes
are estimates the office corrects before building.

**The solver.** Phase 1 (CP-SAT) chooses a time for every session. Hard: no
programme, faculty member or room type double-booked, checked per half of
the semester (an H1 and an H2 course may share a slot), room demand within
supply at every capacity, the free afternoons, hard preferences. Soft:
preferences, electives of one pool kept apart, no evening lectures for first
and second years, no late tutorials, at most two lectures a day per
programme; in a repair, every move costs more than any preference. Phase 2
gives each session its rooms (smallest that fits; the same rooms as before in
a repair). `verify()` then re-checks every hard rule on the result.

On the Monsoon 2026 offerings (150 courses, 40 programmes, 18 pools →
280 sessions, 410 weekly meetings) a build takes about a minute on the
laptop and breaks no hard rule, with no two electives of a pool clashing;
a change such as "SH1 closed on Mondays" re-solves in about ten seconds and
moves only the sessions that have to move.

**Club activities** are checked for club hours (weekday evenings, the free
afternoons), notice (two days; five working days for the auditorium), the
members' programmes being free, and a free lecture room big enough (picked
automatically if none is named). Events over 150 people and the auditorium
go to the office. A request that fails comes back with alternatives that pass,
and the booking rules from the policy documents are cited.

```sh
uv run python -m semester.offerings course-offering/CourseOfferings-M26-V7.pdf --all   # what was read
```

In the web app: **Build timetable** (office: offerings → preferences →
timetable → changes), **Semester preferences** (faculty and class
representatives) and **Club activities** (students; the office approves the
large ones). State is saved in `runs/semester/` (`NTS_SEMESTER_DIR`); the build
time limit is `NTS_SEMESTER_TIME` (default 60 s).

Not modelled: which electives each student actually takes (electives are
only kept apart within a pool), rooms' distance between back-to-back classes,
and the demo personas are not the institute's faculty (preferences name the
faculty member in the text).

## Web app (React + TypeScript front end, FastAPI back end)

```sh
uv run nts-web                       # builds the UI if its sources changed, then API + UI on http://127.0.0.1:8000
uv run nts-web --rebuild             # always rebuild the UI first (--no-build / NTS_BUILD=0: never)
NTS_PARSER=gemini uv run nts-web     # System Two + policy agent on Gemini instead of offline rules
NTS_PARSER=local uv run nts-web      # parser and policy agent on the fine-tuned model (llama-server; NTS_LOCAL_URL, default http://localhost:8080/v1)
uv run nts-web --parser local --test-data 20   # benchmark department; replays 20 held-out test requests (NTS_TEST_DATA)

cd frontend && npm run dev           # hot-reloading UI on :5173, /api proxied to :8000
```

`src/web/app.py` runs the real pipeline on a demo CSE department (8 faculty, 56
sessions) and replays a short history at start-up through it: a conference
absence, preferences, a lab contention that is negotiated, a student's
injection attempt (refused), a lunch-slot request (denied with a citation), a
vague request (clarification), a policy question (answered) and a lab outage
(Lab 3, the only electronics lab, closed in week 9). A room closure is a fact,
not a request: classes that have no other suitable room that week are
cancelled for the week (a make-up is owed), the rest are repaired, and on
approval each affected teacher and section is told
(`pipeline.orchestrator.closures`). Requests run in background threads and the
UI follows them through the event log.

- **Test data (`--test-data N`, or `NTS_TEST_DATA=N`).** Loads the benchmark
  department (`data/synthetic-cse-s0.json`: 30 faculty, 202 sessions) instead of
  the demo one and replays N requests from the held-out test split of
  `data/requests-para.jsonl` (round-robin over the gold actions) instead of the
  demo history. Each case is scored against its gold action; the dashboard
  shows the run and the server prints one line per request.
- **Auth is mocked.** Sign in by picking a person; the client sends `X-User`.
  Every permission is enforced server-side (only the coordinator approves,
  people see their own cases, replies only from the addressee). `?as=F-101`
  on any URL signs in directly, for demos and screenshots.
- **Negotiation is interactive.** When the negotiator needs someone's answer,
  the message waits in that person's inbox (accept an option, counter with
  days and times, or decline). "Let the simulator answer" and per-person
  autopilot let one person demo both sides; unanswered messages escalate at
  the deadline (`NTS_REPLY_DEADLINE`, default 900 s).
- **Replies in your own words.** Under each inbox message a free-text box
  sends the reply to the reply parser (the agent model in `gemini` and
  `local` mode, the keyword handler offline). The sender sees how it was read
  ("You accept option B: ...") and confirms before anything is sent.
- **Human study (`/study`).** Participants enter a code issued on the
  Experiments page (coordinator) and give consent; no name is stored. Team
  raters (`T-n`) rate messages, check explanation claims against their facts
  and label replies, blind to the configuration. Pilot faculty (`P-nn`) rate
  the 24-message pilot set, then use the portal as a demo faculty member
  (default F-102): send a preference and say whether the system understood
  it; in the practice clash a simulated colleague takes the only routers lab,
  the participant asks for it in their own words, answers the negotiator's
  message in their own words, confirms the reading and rates the message.
  Records go to `runs/study/` (`NTS_STUDY_DIR`); the Experiments page shows
  progress, κ and the H3 comparison, and exports everything as JSON. Run one
  pilot participant at a time and reset the demo between them.
- **Demo replays are cached.** `nts-web` keeps CP-SAT answers in
  `runs/solve_cache/`, keyed by the exact model and solver parameters, so the
  start-up history and a demo reset replay in about 25 s instead of about
  2.5 min (LLM answers are cached in `runs/llm_cache/` as before). Simulated
  replies skip their 1.2 s pause while the history is seeded. `NTS_SOLVE_CACHE=0`
  solves afresh; evaluations never use this cache.
- **State survives restarts.** `nts-web` saves each world's state (requests,
  timetable versions, inboxes, simulated people, semester plan) to
  `runs/state/<world>/live.pkl` a few seconds after anything changes, and its
  demo state to `demo.pkl`; the next start carries on from them instead of
  replaying the history. A state saved for a different department, or one the
  code can no longer read, is ignored. A request that was mid-way when the
  server stopped starts again. `NTS_PERSIST=0` replays every time.
- **The model can drop out.** In `local` and `gemini` mode the parser, policy
  check and reply reader fall back to their rule-based stand-ins whenever the
  model does not answer, and go back to the model by itself when it does
  (`src/web/fallback.py`); a model server that is down at start-up does not
  stop the app. The sidebar says which is in use. `NTS_LOCAL_TIMEOUT` (default
  120 s) bounds one model call.
- **Offline by default.** Without an API key, `src/language/rule_parser.py` (a
  rule-based stand-in for System Two: 94% action accuracy and 83% compiled
  constraints on the paraphrased test split, vs Gemini's 100% on validation)
  and `RulePolicyAgent` (real BM25 retrieval, rule-based verdicts) keep every
  downstream layer real.
- A newer request from the same person replaces their earlier constraint on
  the same class ("superseded"), instead of negotiating with themselves.
- The web negotiator disables CP-SAT presolve (it is about 2 of 3 seconds per
  feasibility check at this size) and ranks correction sets before computing
  alternatives, so a negotiation round takes about 15-20 s and a direct repair
  about 3 s. The research runs keep the defaults.

Views: sign-in (persona picker), overview with live activity, new request
with a live lifecycle, requests, case audit (routing, parse, policy with
retrieved/cited rules, the MUS per round, the negotiation thread, claim-to-fact
explanation grounding, diff, fairness, event trace), negotiation inbox,
approvals (coordinator), timetable with versions and rollback, fairness
ledger, handbook with a retrieval playground, how decisions work (tiers,
weights, ladder, safety rules), knowledge graph, observability (LLM quota,
stage latency, routing, faithfulness, safety counters, event stream) and
experiments (results from `runs/`, and the human study), and the study
page for participants (`/study`). Dark and light themes; Ctrl+K palette.

## Key conventions

- **Tier 0** rules (each session once, no room/faculty/group clash, compatible
  room) are built into the model and never switched off.
- Every other **hard** constraint gets an assumption literal. `TimetableSolver.core()`
  and `find_mus()` work only over these, so a conflict is always stated in
  terms of constraints someone owns.
- **Soft** constraints (Tier 5 preferences; `prefer`/`avoid` only) and, in
  repair mode, moved sessions form the objective.
- **Weeks.** `week=None` solves the semester-wide timetable with permanent
  constraints only. `week=7` adds constraints valid in week 7; pass the base
  timetable as `baseline=` so the repair stays local.
- `enumerate_mcs` only drops constraints in `relaxable_tiers` (default Tiers
  3–4). Tier 1–2 options need a human with authority and go through
  escalation instead.

## Next steps

- **Experiment runs (API quota, spread over days):** parsing test split on the
  paraphrased corpus; policy test split; negotiation LLM configurations
  (`evaluation.negotiation --plan`) and the reply benchmark; the safety set with Gemini;
  judge ratings. Each run resumes from the cache after a daily-quota stop.
- **Team:** 100 hand-written requests and two annotators; the institute's
  real handbook; the human study in the web app (`/study`: two team raters,
  5-10 pilot faculty after ethics clearance), then `evaluation.study report`.
- **Local model:** generate the multi-task data (`training.distill`), train it, and score it with the `--local` evaluations.
- **Swaps and hybrid retrieval with the models:** `evaluation.swaps --reader
  gemini`, and `evaluation.policy --retriever hybrid` (Gemini and `--local`)
  to see whether better recall changes the verdicts. The fine-tuned model has
  no swap training data yet; swaps and retrieval-queries.jsonl are
  author-written and should be joined by the team's hand-written requests.
