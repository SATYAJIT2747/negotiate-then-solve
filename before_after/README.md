# Before and after

One folder per phase of the plan to the final submission (15 Nov 2026). In each: the cases (`requests.jsonl`),
the results before the phase's changes (`before.jsonl`) and after them (`after.jsonl`). A measurement a phase only
starts (a new test set) has its own subfolder, with `before.jsonl` now and `after.jsonl` once a later phase changes
what it measures.

## Phase 1: mixed messages, labels, new test sets (3 Nov)

### Chat: a question and a request (or a rule question) in one message: `phase_1/`

26 messages to the side chat (`POST /api/chat`) as Dr. Menon, with `nts-lookup-v3` on the GPU server, the chat
emptied before each. Run with `uv run python before_after/run.py before_after/phase_1 --out before|after`.

| | before | after |
|---|---|---|
| question + request (8) | 0 | 8 |
| several questions + request(s) (6) | 0 | 6 |
| question + rule question (4) | 0 | 3 |
| controls: request, question, rule only (8) | 8 | 8 |
| **all** | **8/26** | **25/26** |

Before, request words sent the whole message on as a request (the questions went unanswered); a rule question
next to a timetable question was dropped. The change (`World.read_message`, `chat_send`'s `_other_parts`,
`ToolAgent.others`): with v3 up, v3 reads every message (it was trained on 150 mixed ones; the request is its
`none` part), the questions are answered, and the rest follows: a rule answer, and one "send as a request?" card
for every request part, which shows what would be sent. With the model down, request words work as before.

Still wrong, for v3.1 / v4 training data:
- P1-17: "How many hours can a teacher teach in a row?" read as `describe`: the rule question is not seen.
- P1-15 passes, but its rule answer is about working hours, not online classes: the rule reader gets the whole
  message ("Friday" pulls in §2.1); asked alone it finds §3.4. The card likewise holds the whole message. Fix:
  each part's own words (`request` on v3.1's `none` parts, `text` in v4).

### Test labels older than v3: `phase_1/labels/`

v3 reads "was my class cancelled?" with `change: cancelled`, "any gaps on Monday?" as `gaps`, "what week is it?"
as `calendar`; the labels were written before those fields and tools. Seven templates in
`data/lookup-test-templates.jsonl` now accept that reading too (`also`). Test set A, v3: **330 -> 346 / 362**
(16 questions; their readings are the same in both files, only the labels changed).

### Long messages: `phase_1/multi/` (new: `evaluation.multi_set`)

200 messages of 2 to 5 parts from the TEST templates only: 1 to 5 questions, 0 to 2 requests, a rule question in
34; 72 joined in ways the training data never joins them (new lines, "Then", "Also,"). Baseline, v3:

| | right |
|---|---|
| whole message (every lookup, and the other parts counted right) | 111/200 |
| every lookup right | 113/200 |
| question parts | 440/561 (78%) |
| requests found (messages with one) | 112/123 |
| by questions: 1 / 2 / 3 / 4 / 5 | 8/12, 45/67, 41/77, 14/36, 3/8 |
| joined as in training / new ways | 75/128, 36/72 |

Misses: a lookup wrong in 62 messages (a few templates again and again: T2-008, T-139's dates, T3-016), the
number of lookups wrong in 25 (15 too many, 10 too few), requests or rule questions counted wrong in 29 (16 a
request missed).

### Memory, then lookup: `phase_1/end_to_end/` (new: `evaluation.end_to_end`)

memory's 400 test conversations through `nts-memory-v1`, then `nts-lookup-v3`. Right when the rewrite is looked
up as the expected rewrite is (the lookups to expect are v3's reading of the expected message: this measures what
memory loses, not v3's own mistakes).

| | right |
|---|---|
| end to end | **387/400** |
| memory alone (memory_set's check) | 385/400 |
| read as typed, without memory | 146/400 |
| misses by stage | memory 12, lookup 1 |

Memory is what makes follow-ups work: without it, 146 of 400 are looked up right. Three of memory's 15 "wrong"
rewrites (kind `list`) are still looked up right.

## Phase 2: the lookup vocabulary, v4 (3 Nov)

Code only: no model yet, so no before and after of a model here (that is Phase 5, v3 against v3.1 and v4 on
every set above). What Phase 2 checks is that v4 can say everything v3 says, with the same lookups.

`agents/lookups_v4.py`: `LookupsV4`, a part per thing a message asks, each tool with only its own fields
(20 kinds of part, at most 8 a message). Shared pieces: `who` (names as written, roles, unknown), `when`
(days, today/tomorrow, dates, part of the day, clock times `at` / `after` / `before`, and the request
vocabulary's `Period`: `next_week`, weeks, from/to), `room`, and `which` for the class `move_options` moves.
`request` and `rule` parts carry their own words; `other` is left out. The code works out slots and week numbers
(`language.intents.slots` / `weeks`), then v3's checks run unchanged (`to_choice` gives a `ChoiceV3`), so
`Answerer` and every test set treat v3 and v4 alike. `LOOKUP_SCHEMA = 4`.

v3's training rows in v4 (`training/lookup_data_v4.py`, `data/lookup-train-v4/`): each row's v4 answer is read
by `ToolAgent` next to its v3 answer, and kept only if the lookups are the same.

| | v3 rows | in v4, same lookups | left out |
|---|---|---|---|
| train | 2,503 | 2,399 | 104 |
| val | 495 | 459 | 36 |
| distilled (Gemini's rewordings) | 583 | 571 | 12 |

Every row left out is a mixed message: v3's "none" didn't keep the request's words, so its v4 `request` part
can't be written. Phase 3 writes those messages again, with the words, for v3.1 and v4 alike. No row gives other
lookups. v4 rows are about a third shorter (prompt 3,805 characters against 5,786; answers about the same), so
training takes less time.

Found on the way, and fixed in the schema: `free_rooms` and `who_is_free` had no `unknown` ("is the auditorium
free?" would have listed lecture rooms).

In the chat with v4 (`tests/test_lookups_v4.py`): a rule question inside a mixed message is answered from its own
words (P1-15's "online classes" now finds §3.4, not §2.1), and the request card holds only the request.

## Phase 3: training data for v3.2 and v4 (3 Nov)

`python -m training.lookup_data_v3_2` writes `data/lookup-train-v3_2/` and `data/lookup-train-v4/`: **the same
rows** in the two formats (each row checked in both; kept only if both give the expected lookups), so the two
Kaggle runs differ only in the format. (`nts-lookup-v3_2`, not v3_1: that name is v3's second session, on the
server.)

| | rows | of which new |
|---|---|---|
| train | 2,354 | 655: 410 mixed, 301 long, 256 slips (with validation's share) |
| val | 768 | 312 |
| distilled (Gemini's rewordings of v3's rows) | 571 | 0 |

New: mixed messages of 1 to 4 questions with 1 to 3 requests (806 request parts, 145 rule questions, 66
"thanks"), each request and rule question with its own words; long messages of 4 to 8 questions (v3: at most
5); 32 new training templates for the slips (`data/lookup-v3_2-train-templates.jsonl`: "a 2 hr slot", "is Rao
away", "the exam hall", "any section free for a guest session", "list the teachers", "when are we and Dr X both
free", rule questions on hours in a row), none of them a test template's wording. v3's rows replayed (1,699 of
2,399), its mixed messages left out (no request words). Left out of both formats: 4 rows whose two readings
differ (2 "tomorrow" asked on a weekend, where v3's label has the wrong week; 2 long messages splitting a
"separately" question differently).

Notebooks: `notebooks/train_lookup_v3_2_kaggle.ipynb` and `train_lookup_v4_kaggle.ipynb`, both made from v3's by
`scripts/make_lookup_notebooks.py` (only the header, settings, row check, quick check and manifest differ; v4's
row check takes each tool's fields from the schema). Both continue from v3's adapter at v3's learning rate.
Each notebook's row check passes every row of its data here (0 left out).

Not yet: Gemini's rewordings of the new rows (needs the paid key's OK).

### Memory v1.1 (4 Nov)

`python -m training.memory_data_v1_1` writes `data/memory-train-v1_1/`; v1's data and its 400-conversation test
set are unchanged (v1's numbers stay comparable). New kind `plural`: a plural word means all of it: "these days"
(every day just asked about), "those times" (every free span the reply gave; "then" after several is still asked
back), "the same time" (another day, next week, someone else). Also more of v1's weakest kinds (`list` 9/14,
`two_back` 9/12, `ask` 12/16 on its test set) and `same` rows, so a fifth of the rows are still messages to leave as
they are (v1: 24%).

| | rows |
|---|---|
| train | 1,850 (900 new: 512 plural, 223 same, 135 list, 67 two_back, 63 ask; 950 of v1's) |
| val | 380 |
| distilled (Gemini's rewordings of v1's rows) | 400 |
| test-plural (test templates and wordings only) | 64 |

Notebook `notebooks/train_memory_v1_1_kaggle.ipynb` (from v1's by `scripts/make_memory_notebook.py`): continues from
nts-memory-v1's adapter at 5e-5; ~3 h. Its row check passes every row here.

Before: memory v1 on test-plural (`phase_3/memory-plural-v1.jsonl`): **52/64**. By kind: "these / those / same
days" **26/26**, "those times / slots / hours" **17/17** (v1 already does both: the known gap was overstated), "the
same slot / time for X" **9/21**: v1 keeps the earlier person and adds the new one ("is Dr. Reddy free on Wed at 9 am
with Section 4?") instead of putting the new one in their place ("is Section 4 free on Wed at 9 am?"). That is what
v1.1's "same time" rows teach; the other two kinds check it loses nothing.

## Phase 5: v3 against v3.2 and v4 (4 Nov)

On the GPU server: `nts-lookup-v3_2` (session 1 of 2: about 90% of its steps) and `nts-lookup-v4`. The server's LAN
cable drops for moments, in bursts: the scripts that call it directly wait and try again; `before_after/run.py`
(the chat, through the app, which falls back to the regex at the first drop) now sends a message again when it was
answered without the model (`… sent again after the server dropped`), and every result below was checked to be the
model's (the reader of every reply).

| | v3 | v3.2 (session 1) | v4 |
|---|---|---|---|
| Phase 1 chat cases (`phase_1/v3_2.jsonl`, `v4.jsonl`) | 25/26 | 25/26 | **26/26** |
| memory v2 chat cases (`memory_v2/`, memory v1) | – | 5/8 | **7/8** |
| long and mixed messages (`phase_5/multi-*.jsonl`): whole message | 111/200 | 119/200 | **140/200** |
| … question parts | 440/561 | 445/561 | **483/561** |
| … requests and rule questions counted right | 171/200 | 177/200 | **191/200** |
| … requests found (of 123 messages with one) | 112 | 117 | **122** |
| … 4 questions / 5 questions | 14/36, 3/8 | 15/36, 2/8 | **23/36, 5/8** |
| … joined in ways training never joins them | 36/72 | 42/72 | **52/72** |
| test set A (`phase_5/A-*.jsonl`) | **346/362** | 339/362 | 337/362 |
| test set C, names shuffled (`phase_5/C-*.jsonl`) | **346/362*** | 334/362 | 340/362 |
| test set D, shared courses, never seen (`phase_5/D-*.jsonl`) | **343/360*** | 335/360 | 333/360 |

| memory then lookup, end to end (`phase_5/e2e-*.jsonl`, memory v1) | 387/400 | 386/400 | **388/400** |
| … misses by stage | memory 12, lookup 1 | memory 12, lookup 2 | memory 12, lookup 0 |

\* v3's C and D runs (`runs/lookup-v3-{C,D}.jsonl`) were scored before the labels' `also`: counted again with
today's labels (a reading right if v3.2 or v4 was right with the same reading), at least 346 and 343 (332 and
327 as first scored).

v3.2's misses on the chat cases: P1-05 ("I want my lecture moved…" read as a what-if). Memory v2: v4's request
parts carry the words memory needs ("Could she take my IS lecture then?" → "Could Dr. Iyer take my IS lecture on
Friday at 11 am?"); v3.2's card held the whole message there, and memory dropped the time in M2-01 ("to Friday").
Both asked back with two free spans ("Which time do you mean: 9 am–12 pm or 2–5 pm?", the code's check). v4's one
miss: its request part took in the question ("Am I free on Friday morning? If so, shift … to then").

Test set A, against v3: v4 lost 25 questions and gained 16, v3.2 lost 19 and gained 12. Gained, by both: every
slip the new templates teach (a 2 hr window, on leave, the exam hall, a guest lecture, the faculty, when we are
free). Lost, and why:

| lost | v4 | v3.2 | why |
|---|---|---|---|
| "in week 2" read as week 1 | 6 | 2 | no "in week N" in v3.2/v4's data (v3 knew it from v2's): forgotten |
| calendar dates | 0 | 6 | the same, for dates |
| "who could take my class?": course "me" | 3 | 1 | the model writes course "me" (the check should drop it) |
| "can I have Thursday free?" read as a question | 3 | 3 | a request read as a question (the chat still offers the card: request words) |
| "why was X's class cancelled?" read as changes | 3 | 0 | private, not looked up |
| "can I book Room 102 on …" read as free_rooms | 3 | 0 | a request read as a question (card still offered) |
| "who else teaches X and is free" read as free_hours | 3 | 0 | the other tool |
| "how many teaching hours does X have" read as counts | 2 | 0 | v4 is right: the label predates counts |
| other | 2 | 7 | |

C and D: the same losses, in other departments (v4 lost 22 and gained 16 on C, lost 27 and gained 17 on D): week N
read as N−1, course "me", "can I have … free", "can I book …", "why was … cancelled", "who else teaches …", plus
"is it possible to shift my X to Monday" read as move_options (1 on C, 2 on D). Not memorised: forgotten, the same
everywhere.

Next: the check drops a course "me" (code), the label accepts counts, and a v4.1 top-up (from v4's adapter, with
replay) teaches "in week N", dates, "can I have … free" and "book …" as requests, and "why …" as private.

## Memory v2: a request that leans on the answer in the same message (4 Nov, code only)

"When am I free on Friday? Move my DS lecture **there**." Memory works between messages; here "there" is the
answer to the message's own question. With v3.2 or v4 (a request part keeps its words), the chat answers the
questions, then gives memory the questions and their answer as the chat just before the request, and memory does
what it was trained to ("move my DS lecture there" after "You are free on Friday 2–3 pm" is "move my DS lecture to
Friday 2–3 pm"): one time: the card holds the request in full (and `said`, as written); several: the question
back ("Which time do you mean: 9–11 am or 2–4 pm?"), no card, and the answer to it ("2–4 pm") is read as a
follow-up, memory seeing what was asked about (`About “move my DS lecture there”: …`). Memory off, or nothing to
fill in: the request as said. No new adapter: memory v1 (v1.1 when trained). With v3 (no request words) nothing
changes. `tests/test_memory_v2.py` (fake models).

Considered and left out: memory answering with an edit in v4's vocabulary instead of a sentence (one model call
fewer, checkable), because memory would then have to know every other adapter's format (lookups, the team's
request vocabulary, rule questions), against the split. Later (memory v3): a structured state of what the last
answers held, given to memory with the chat, memory still answering in words.

`memory_v2/requests.jsonl`: 8 chat cases (Dr. Menon, week 15: Friday has one free span, Monday and Wednesday
several), scored with `card_has` / `card_lacks` / `asks`; run once v3.2 or v4 is served:
`uv run python before_after/run.py before_after/memory_v2 --out after --lookup nts-lookup-v4` (`--out before`
without `--lookup`: v3 as the app has it now, which can't fill anything in).
