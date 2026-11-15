# Encoder intent reader (GLiNER-style): design sketch

Idea from the "encoder decision models" talk: instead of a decoder LLM generating the intents JSON token by token,
put the **labels themselves into the input** as special tokens, run one encoder pass, and read every decision off
the contextualised embeddings of those label tokens (classification) or by matching label embeddings against token
spans (extraction). Models in this family: GLiNER, GLiClass, GLiNER2 (Fastino), all ~100-300M parameters and CPU-capable.

## Where it fits

It does not replace the pipeline. It slots into **step 2 (Parse)**, which already has the right shape
(proposal brief: router -> LoRA compiler for routine requests, base model for ambiguous ones; `language/system_one.py`
has the `p >= tau` fast path). The encoder becomes the routine path; Qwen3.5-4B + intents LoRA becomes the fallback.

```
request ─► Encoder pass 1 (action, intent triggers, composition) ─┐
                                                                  ├─► normaliser ─► IntentOutput JSON ─► checks.py ─► resolver ─► solver
           Encoder pass 2 (per trigger: argument spans + attrs) ──┘          │
                 any gate fails ─────────────────────────────────────────────┴─► Qwen intents LoRA (today's path)
```

Output contract stays `IntentOutput` (`src/language/intents.py`), so `checks.py`, the resolver, the suites and the
head-to-head harness score it unchanged.

## How much it can cover (measured on `data/compiler-intents`)

| split | single intent | multi, flat | nested (any_of/ranked/fallback) | refers back (`which.same`) | non-compile actions |
|---|---|---|---|---|---|
| train (4623) | 3671 (79%) | 293 (6%) | 114 (2.5%) | 19 | 526 |
| test (417) | 319 (76%) | 0 | 0 | 0 | 98 |

Course names in the gold JSON appear verbatim in the message 1059/1081 (train) and 68/68 (test), so span extraction
is a natural fit. Nested and refers-back cases go to the LLM by design.

## Special tokens

| token | meaning |
|---|---|
| `[CTX]` | start of context header (role, current week, today) |
| `[MSG]` / `[/MSG]` | the message (data; never instructions) |
| `[P]` | a task prompt: the task's name follows; its embedding is the task query |
| `[L]` | a class label for a classification task (softmax or sigmoid over the `[L]` embeddings of that task) |
| `[E]` | an entity/argument label for span extraction (scored against every candidate span, max width ~8 tokens) |
| `[D]` | a short description of the preceding `[L]`/`[E]` (the talk's "description" trick; matters for role labels) |
| `[T]` / `[/T]` | marks the trigger span of the intent being filled in pass 2 |
| `[SEP]` | separates blocks |

## Pass 1: what the message is

```
[CTX] faculty [SEP] week 6 [SEP] Wed [SEP]
[MSG] Can't take my Tuesday 2pm DBMS lecture next week, could we move it to Thursday morning? I have a conference. [/MSG]
[P] action      [L] compile [L] clarify [L] investigate [L] answer [L] out_of_scope [L] withdraw      (softmax)
[P] composition [L] independent [L] any_of [L] ranked [L] fallback                                    (softmax)
[P] trigger
  [E] move         [D] one of the sender's classes should go to another time
  [E] unavailable  [D] the sender cannot teach at some time
  [E] avoid        [D] the sender would rather not teach at some time
  [E] only         [D] the sender wants to teach only at some times
  [E] cancel       [D] a class should not happen
  [E] extra_class  [D] an additional class should be scheduled
  [E] swap         [D] exchange a class slot with a colleague
  [E] relate       [D] two classes should be placed relative to each other
  [E] limit        [D] a cap per day or week, or a gap
  [E] load_limit   [D] no more than N classes in a row
  [E] needs_room   [D] a class needs a room with equipment
  [E] room_closed  [D] a room cannot be used
  [E] club_booking [D] a student club wants a room
  [E] clash_report [D] two classes clash for students
```

Each trigger span found = one intent instance (two "move" spans = two moves). Here: `action=compile`,
`composition=independent`, trigger `move it to` -> `move`. (The "can't take" is the reason for the move, not an
`unavailable` intent; the training labels teach that, the same way the LoRA learned it.)

## Pass 2: fill one intent (one pass per trigger, batched)

Each kind has its own argument schema; role labels make source vs target explicit, which plain NER cannot.

```
[CTX] ... [MSG] ... could we [T] move it to [/T] Thursday morning ... [/MSG]
[P] move
  [E] class course   [E] class day   [E] class time   [E] class kind
  [E] target day     [E] target time [E] target after [E] target before
  [E] week [E] from week [E] to week [E] except week
  [P] target part    [L] morning [L] afternoon [L] lunch [L] none
  [P] period         [L] this_week [L] next_week [L] rest_of_semester [L] none
  [P] parity         [L] odd [L] even [L] none
  [P] firmness       [L] hard [L] soft
  [P] justification  [L] none [L] stated [L] verified
```

Spans: `DBMS`->class course, `Tuesday`->class day, `2pm`->class time, `lecture`->class kind, `Thursday`->target day.
Classes: part=morning, period=next_week, firmness=soft, justification=stated.

Per-kind extras: `relate` adds `[E] other course/kind` and `[P] relation` (the 7 `Relation` values) + `[P] negate`;
`limit` adds `[E] number` + `[P] measure` (4 `Measure` values); `swap` uses `mine ...`/`theirs ... owner`;
`club_booking` uses `club, activity, day, start, end, attendees, room, date` + `[P] room kind`;
`needs_room` adds `[E] equipment`; `clash_report` uses `[E] course` (multi) + day/time.

## Normaliser (deterministic, no model)

Span text -> typed values: `Tuesday`/`tues`->`Tue`; `2pm`/`14:00`/`2 o'clock`->`14:00`; `weeks 9-11`->from/to;
`lecture`/`lab`->`Kind`. Reuse `language/lexicon.py`, `shorthand.py`, `names.py`. A span that fails to normalise
is a gate failure (fallback), never a guess.

## Gate (when the encoder's answer is used)

Accept only if all hold, else send the request to the Qwen LoRA:
1. action and every used class/span probability >= tau (tune with the existing `tune_tau`, target: precision on
   accepted rows >= the LoRA's accuracy on the same rows);
2. `composition == independent` and no `which.same`;
3. every span normalises, `IntentOutput` validates, `checks.py` and the resolver accept it.

## Training data (no new collection)

**Balanced by complexity** (`python -m training.encoder_data`, recipe `data/recipes/encoder-intents-v1.toml`,
output `data/encoder-intents/`, report in its `manifest.json`):

| tier | v13 train | encoder train (7482) |
|---|---|---|
| non_compile | 526 (11%) | 1125 (15%) |
| single | 3446 (75%) | 3000 (40%), water-filled over kinds and generators |
| multi (2-3 independent) | 290 (6%) | 1861 (25%) |
| nested (any_of/ranked/fallback) | 114 (2%) | 881 (12%) |
| linked (relate / `same`) | 247 (5%) | 615 (8%) |

Complex rows join the single-intent generators' own training wordings (so a kind reads the same inside a long
message as alone; see `training/balance.py` for why), each part resolved alone, the whole compiled, then the
repo's quality gates (rows failing the firmness rubric because one part's hedge reads as the whole message's
are dropped: 25). Held-out wordings + held-out connectors give `composed.jsonl` (294 rows), a suite for the
complex tiers; the frozen test split and `data/suites` are untouched.

**Span labels** (`python -m training.encoder_spans` -> `data/encoder-intents/spans/{train,val,composed,test}.jsonl`
+ `report.json`). Each gold value is looked up in the message (days incl. typos, times incl. ranges and bare
hours, week numbers and ranges, number words, class kinds, course titles and their abbreviations via
`language/shorthand.py`, dates, names). Mentions are chosen for all intents together (beam search: tightest
stretch per intent, penalties for reusing another intent's words or going backwards); a day/week inside a said
range ("Wednesday through Friday", "weeks 9-11") is covered by the range ends; a class kind never said beside
its day+time (a swap's "slot") is `inferred`, not pointed at.

| split | aligned |
|---|---|
| train | 7388/7482 (99%) |
| val | 743/749 (99%) |
| composed | 294/294 (100%) |
| test (frozen) | 417/417 (100%) |

Precision check (fresh composed messages with known part boundaries): 0 of 4437 spans land outside their own
part. Of 42 moves that say one value for both the class and its target, all 42 assign source and target the
right way round. What stays unaligned is truly implicit ("move it to another day, just not Friday" -> the
other days, 135 values from `exception`; "Tuesday morning" -> before 12:00): those rows are the LLM's.
Non-compile rows train pass 1 only.

## Model

- **GLiNER2-base** (Fastino, ~200M): one schema covers classification, span extraction and per-instance structures,
  i.e. pass 1 + pass 2 natively. Closest to the talk. (Its fine-tuning API needs checking before committing.)
- Fallback: **DeBERTa-v3-base or ModernBERT-base** with the tokens above and our own heads (span scorer = MLP over
  [start; end] dot `[E]` embedding; classifier = MLP over `[L]` embedding). More code, full control.

4.6k rows at ~200M params trains in minutes on one T4; inference runs on CPU.

## Evaluation (fits the no-ablation plan: one model, one run, release rule)

Same scorers as today, on test (417), `data/suites`, head-to-head (41):
- **coverage**: share of requests the gate accepts;
- **accuracy on covered** vs the LoRA on the same requests;
- **end-to-end** (encoder + LoRA fallback) vs v13b-e2 alone — ship only if no suite drops;
- **latency / hardware**: CPU ms per request vs Qwen 4B on GPU.

Side use: the pass-1 head (question vs change, multi-label) can replace System One in `harness/pick.py`, which
today needs the `CHANGE` regex to patch cases like "when am I free? I want to take an extra class".

## Risks

- Implicit values ("move it to the usual slot", "same as last week") have no span: gate catches them via
  normalise/validate failure, but coverage drops.
- Negation and scope ("not before 10 except Fridays") are where the LoRA earned its accuracy; watch avoid/only/
  except_week suites.
- Paraphrase alignment may lose a slice of training rows; report it.

## Results, run 1 (2026-11-10, Kaggle T4; `kaggle-results/nts-encoder-v1`, `nts-gliner2-v1`)

Two readers on encoder-intents-v1: **DeBERTa** (`kaggle/encoder/nts_encoder.py`: deberta-v3-base, our `[P]/[L]/[E]`
heads, 4 epochs) and **GLiNER2** (`nts_gliner2.py`: fastino/gliner2-base-v1 fine-tuned; classifications and
structures as separate examples, since a structure's choice labels are prepended to the text and become a
shortcut otherwise). Zero-shot GLiNER2 reads 2.6% of test.

**Outcome match** (`python -m evaluation.encoder_outcome --preds <run>/predictions --llm <misses>`): predictions
normalised to `IntentOutput` and scored exactly as the LLM (`evaluation.intents`, with `language.checks`). The
normaliser's ceiling (gold spans decoded) is 100% on test and every suite but test-typo (99.5%) and val (99.2%).

| | DeBERTa | GLiNER2 | v13b-e2 (Qwen 4B LoRA) |
|---|---|---|---|
| test (417) | 89.4% | **94.7%** (silent 0.2%) | 96.4% |
| test-noisy / test-typo | 78.7 / 76.5% | 86.8 / 87.3% | 94.5 / 95.0% |
| val (749) / composed (294) | **98.1 / 58.2%** | 92.7 / 71.8% | - |
| ms per message (T4) | 83 | 25 | ~1000+ |

Suites where an encoder is below the LLM: exceptions (D 0%, G 50% vs 97.5%), answered (57.5/32.5 vs 97.5),
clubs (22.5/75 vs 100), choices (50/50 vs 100), periods (G 65), week_patterns (50/67.5 vs 82.5). Above it:
same_day_part (D 70, G 100 vs 55).

**Gates and the hybrid** (encoder where the gate accepts, the LLM elsewhere), test:

| gate | coverage | accuracy on covered | hybrid | LLM alone |
|---|---|---|---|---|
| DeBERTa conf >= 0.98 | 76.5% | 93.4% | 94.0% | 96.4% |
| GLiNER2 conf >= 0.98 | 99.5% | 94.9% | 94.7% | 96.4% |
| both decode to the same reading | 85.6% | 96.1% | 95.7% | 96.4% |

GLiNER2's confidence is saturated; DeBERTa's is informative; agreement is the best gate. The rows both readers
get wrong alike are mostly one action: 14 of 20 are the test's held-out withdraw wording ("Never mind my message
from yesterday; please leave my timetable as it is."), read as a question; v1 trained withdraw on four wordings.

## Run 2 (encoder-intents-v2)

`data/recipes/encoder-intents-v2.toml`: v1 plus `withdraw_more` (`training/encoder_data.py`, 12 new training-only
withdraw wordings, none the test's; 130 of the 281 withdraw rows). Two kernels (`kaggle/encoder/make_kernel.py`):
`nts-gliner2-v2` (GLiNER2 on v2) and `nts-encoder-v2` (our DeBERTa heads started from GLiNER2's encoder and its
trained `[P]/[L]/[E]` embeddings: GLiNER2's reading of unseen wordings with our heads for several requests).

### Results, run 2 (2026-11-10; `kaggle-results/nts-encoder-v2`, `nts-gliner2-v2`)

`nts-encoder-v2` loaded GLiNER2's `[P]`/`[L]`/`[E]` embeddings (only `[CTX]`, `[MSG]`, `[/MSG]` new); 4 epochs,
45 min on a T4, best epoch 4 (val exact 0.961). Outcome match, scored as the LLM:

| | DeBERTa v1 | **DeBERTa on GLiNER2 v2** | GLiNER2 v1 | GLiNER2 v2 | v13b-e2 |
|---|---|---|---|---|---|
| test (417) | 89.4% | **98.1%** (silent 1.2%) | 94.7% | 97.6% | 96.4% |
| test-noisy / test-typo | 78.7 / 76.5 | 89.0 / 90.9 | 86.8 / 87.3 | 88.7 / 87.5 | 94.5 / 95.0 |
| val / composed | 98.1 / 58.2 | **98.4 / 76.9** | 92.7 / 71.8 | 94.5 / 67.0 | - |
| ms per message (T4) | 83 | 83 | 25 | 26 | - |

Starting from GLiNER2's encoder gave our heads its reading of unseen wordings (exceptions 0 -> 77.5%, clubs
22.5 -> 82.5%, week_patterns 50 -> 82.5%, same_day_part 100% vs the LLM's 55%) and kept the complex tiers. The
withdraw wordings fixed the test's held-out withdraw (action accuracy on test 92.3 -> 99.3%).

**Hybrid** (`--agree` the other v2 reader, `--tau 0`; a reading that does not decode always goes to the LLM;
`--llm-choices`: any reading with a choice goes to the LLM):

| | coverage (test) | accuracy on covered | hybrid test | noisy | typo | suites below the LLM |
|---|---|---|---|---|---|---|
| DeBERTa-v2 conf >= 0.98 | 85.9% | 99.2% | 98.1% | 91.1 | 94.5 | answered, choices, moves, noisy, typo |
| both v2 agree | 92.6% | 99.2% | 98.3% | 94.5 | 96.2 | answered (72.5 vs 97.5), choices (52.5 vs 100), clubs (97.5 vs 100) |
| both agree, choices to the LLM | 92.6% | 99.2% | **98.3%** | **94.5** | **96.2** | answered (72.5 vs 97.5), clubs (97.5 vs 100) |

Both readers agree wrongly on two held-out wordings: "X, or failing that Y" read as `any_of` (gold `ranked`),
hence choices to the LLM; and `answered` ("one of the days in week 7. Oh, Friday."), read as a question. Periods
(50%) is a model error both make: "over the coming two weeks" read as rest_of_semester with a count, which does
not validate, so the gate sends it to the LLM (hybrid 100%).
