# Improving the agentic app: a running log

What we planned, why, and what happened after we built it. One section per step, newest at the bottom.

## The plan

| Step | What | Why |
|---|---|---|
| 0 | Make the model server fast | One question took 110 s |
| 1 | Make it stop failing: send only the tools a request needs, retry on server errors, save requests to disk | Every call sends all 29 tools (~6,300 tokens of an 8,192 budget); one server error ends the request; a restart loses every request |
| 2 | Chat threads with memory | Each request starts from zero, so "also do Thursday" or "undo that" can't work |
| 3 | A readable "thinking" panel | The steps are shown as raw tool names and JSON |
| 4 | Better human in the loop: confirm before submit, before/after timetable, stop, edit and resend | The person only sees the result at the end |
| 5 | More caching | The same lookups and draft checks are repeated |

---

## Step 0: the model server was slow (done, 2026-10-31)

**What we saw.** "What are my classes tomorrow" took 110 s. The first model call alone took 100 s.

**Why.** The llama server loaded all 27 LoRA adapters (~1.1 GB) next to the 4B Q8 model (~4.5 GB) on a 6 GB
RTX 4050. The GPU was full (5,640 of 6,141 MB), so Windows spilled into normal RAM, which is very slow. The
server also had one 8,192-token memory shared by 4 slots, so two requests at once ran out of room ("Context
size has been exceeded").

**What we changed.** Restarted the server with only the adapter the app uses, and 2 slots with their own
8,192 tokens each:

```powershell
cd "C:\Users\satya\Desktop\LLM Serve"
.\llama.cpp\llama-server.exe -m models\Qwen3.5-4B-Base-Q8_0.gguf --lora-scaled loras\nts-agent-v6-dpo.gguf:0 --jinja -ngl 99 -c 16384 -np 2 --cache-ram 1024 --host 0.0.0.0 --port 8080
```

**What happened.**

| | Before | After |
|---|---|---|
| Reading the prompt | 25 tokens/s | 1,393 tokens/s |
| Writing the answer | 5.6 tokens/s | 29 tokens/s |
| "What are my classes tomorrow", end to end | 110 s | 11 s |
| Context per request | 8,192 shared by 4 | 8,192 each, 2 at a time |

The answer was right: two Computer Architecture lectures on Monday, 2 to 3 pm in Room 103 and 4 to 5 pm in
Room 104.

**Still to know.** `serve.ps1 -Lora` still loads every adapter. Use the command above for the app, or load only
the adapters an eval needs.

---

## Step 1: make it stop failing (done, 2026-10-31)

### What we planned, and why

1. **Send only the tools a request needs.** Every model call sent all 29 tools, about 6,300 tokens of an
   8,192-token context. That left little room for the conversation, and a long request ran out of space.
   Plan: let System One (the classifier) read the request first and pick the tools.
2. **Retry instead of failing.** One busy-server error or one "context size exceeded" ended the whole request.
3. **Keep requests when the server restarts.** They lived only in memory, so a restart lost all of them.

### What we found before building

- The `agentic` branch did not use System One at all, and the SVM work was on another branch. We copied just
  that one commit onto a new local branch, `agentic-improvements` (not the 62 other `simplify` commits).
- **System One could not recognise questions.** "When am I free on Monday?" came out as a *preference*,
  because it was never taught a "question" label. For change requests its labels were too noisy to choose
  narrow tool sets: a load limit, a "keep a gap between classes" or a "right after the lecture" request landed
  on various labels, so a narrow set would often miss the tool the agent needs.
- So the design became: **teach it one new label, `question`. A question it is sure of gets only the tools
  that read. Everything else gets every tool.** Sending a change too few tools would stop it; sending a
  question too many only costs tokens.

### What we built

| Part | Where |
|---|---|
| The new `question` label, learnt from 40 handwritten questions (`data/questions.jsonl`) and half the handbook questions (`data/retrieval-queries.jsonl`) | `src/language/system_one.py`, `src/language/corpus.py` |
| The tool picker: a question with confidence of at least 0.7 gets the read, rules, talk and end tools (~2,300 tokens instead of ~6,000); anything else gets all tools | `src/nts/harness/pick.py`, `src/nts/harness/loop.py` |
| A tool left out by mistake still runs if the model calls it, so a wrong pick can never block a change | (the harness checks the agent's own permissions, not the pick) |
| Retry: a busy or unreachable server is tried 2 more times, waiting 2 s then 4 s. A full context trims old tool results and asks once more, and the page doesn't show it as a failure | `src/nts/models/openai_chat.py`, `src/nts/harness/loop.py`, `src/nts/api/service.py` |
| On start, the server rebuilds this department's earlier requests from the telemetry log. A request that was waiting or not yet approved shows as `interrupted` (its draft lived only in memory) | `Service.restore` in `src/nts/api/service.py` |
| `--pick-tools` switch on `ask`, `eval` and `serve` | `src/nts/__main__.py` |

### What happened

**Choosing the confidence threshold.** We tested on wording the classifier never saw: 70 questions, and 390
change requests from 2 departments.

| Threshold | Questions that got the short tool list | Changes that wrongly got it |
|---|---|---|
| 0.5 | 61 / 70 | 9 / 390 |
| 0.6 | 55 / 70 | 0 / 390 |
| **0.7 (chosen)** | **47 / 70** | **0 / 390** |
| 0.8 | 31 / 70 | 0 / 390 |

No change request scored above about 0.59, so 0.7 keeps a safety margin.

**The agent, before and after** (the local model with the `nts-agent-v6-dpo` adapter, on the demo department).

| | Baseline (all tools) | With tool picking |
|---|---|---|
| All 42 scenario families, 2 each | 78 / 84 (93%) | 78 / 84 (93%) |
| Question families only (q_free, q_classes, rules_question; 4 each) | 10 / 12 | 10 / 12 |
| Tokens on those 12 questions | 154,564 | 92,488 (**-40%**) |
| Time on those 12 questions | 61.5 s | 72.7 s |
| "Context size exceeded" errors | 0 | 0 |

- The same 2 rules questions failed in both runs: the agent declined instead of answering. That is the agent's
  own mistake, not the picker's.
- Change requests get the same tools as before. The few results that differ between the two runs (`clarify`,
  `move_apart`) are run-to-run noise from the server.
- **Fewer tokens did not make it faster yet.** The server caches the start of the prompt (system prompt and
  tools). A different tool list can't reuse that cache, so its first call reads the whole prompt again. Step 5
  (caching) should look at this.

**Tests.** 14 new tests for the picker (`tests/nts/test_nts_pick.py`) and 7 for retry and restart
(`tests/nts/test_nts_reliable.py`), all passing. All 152 `nts` tests pass.

### Still to know

- The 6 failures in the full eval are the agent's reasoning, not this step's: `clarify` moves the class in
  week 7 too, `no_reply` submits without the teacher's answer, `or_windows` gets stuck, `after_lecture`
  declines, `move_apart` doesn't make the change. Worth new training data later.
- Some "handbook questions" in `data/retrieval-queries.jsonl` are really requests ("I want to reserve the big
  hall for the coding club meetup"). Cleaning them would make the question label sharper.
- On this branch's intent data, the classifier's "new kinds sent down the fast path" rate is 28% (limit 25%), so
  that slow test fails here. On the `system-one-svm-new-labels` branch it was 21%.

---

## Steps 2 and 3: a chat with memory, a "thinking" panel, and an inbox for the office (done, 2026-10-31)

You asked for a ChatGPT-style page instead of one-off requests. Steps 2 (memory) and 3 (thinking panel) were
built together, along with the problems you found while testing.

### What we planned, and why

| You saw | Why it happened | What we changed |
|---|---|---|
| Each request started a new thread; a follow-up didn't know what you'd said | Every request was a separate agent run that started from nothing | **Chats.** A chat holds several messages. Each new message gets the chat's last 4 messages and what came of them (the answer, and any change drafted) as memory |
| No way to see what it was doing, except raw tool names and JSON | The page listed every step as it ran | **A "Thinking" panel** on each answer, open while it works, folded after. Each step in plain words ("✓ Checked free hours · Monday"), with the raw call one click further |
| To answer the agent you used a separate box | Answers went through a separate form | **Answer in the chat.** If the agent is waiting for *you*, your next message is the answer |
| To answer you had to type | Only a text box | **Answers to pick.** The page reads options off the question ("Wednesday 9 am, 10 am or 11 am?" → three buttons and *None of these*; "(lecture or practical)" → two; a yes/no question → Yes / No; "which class?" → your classes). Typing still works |
| Changing your mind left the old draft waiting for approval | A draft stayed "submitted" until the office decided | A new message in the chat **replaces** an undecided draft, so the office can't approve the old one by mistake |
| The agent got stuck in a loop ("make_offer" again and again) | It asked a teacher to give way before drafting and checking anything, and the refusal only said "no blocking rule 1" | The refusal now says the order: draft the change, then `check_draft`, then make the offer. A stuck request now ends with a plain message asking for the missing detail (which class, which week) instead of "stuck · 3 repeated calls" |
| "Which kind and which section?" when Dr. Khan teaches Computer Graphics to only one section | The tool asked for both whenever anything was unclear | It names only what differs: "Computer Graphics has the 1-hour lecture and the 2-hour practical (all for G-03)" |
| The office could read every teacher's chats | The office saw everything | The office sees **only what reaches it**: changes sent for approval (and what came of them), and requests the agent passed on |
| The office got a chat window | It used the same page as teachers | **An inbox for the office.** Under *Requests*: *Needs you* and *Done*, one item per request. Open one to see the request, the agent's summary, the proposed changes and how the agent got there, then **Approve**, **Reject**, or **send a message** to the teacher (optionally with Approve or Reject). The message shows up in the teacher's chat. The office's own chat moved to *Agent chat* |
| Signing in as the office in a second tab switched the first tab too | One sign-in cookie for the whole browser | **Each tab signs in on its own** (its session is kept per tab), so one tab can be Dr. Menon and another the office |
| The page went blank and said "Not Found" once | The page file changed while the old server was still running | Not a code problem: it went away after the restart |
| The sidebar was always open, and scroll bars always showed | | **☰** hides and shows the sidebar (remembered). Scroll bars appear only when the pointer is over the area |

### What we built (where)

- Memory and threads: `src/nts/harness/loop.py` (`history` and `thread` on `run`) and `src/nts/api/service.py` (`start(thread=...)`, `thread`, `threads`, `remembered`)
- API: `GET /api/threads`, `GET /api/threads/{id}`, `thread` on `POST /api/episodes`, `POST /api/episodes/{id}/note`, a note on approve, `GET /api/office/requests` (`src/nts/api/__init__.py`)
- Answers to pick: `src/nts/api/choices.py`
- The office's messages are recorded as `note` spans, so they outlive a restart like everything else
- The page: `src/nts/api/static/index.html` (chat, thinking panel, office inbox, picks, per-tab sign-in, sidebar)

### What happened

- **A real conversation with the local model** (Dr. Rao): "what subjects do I teach?" → "Digital Logic, Machine Learning, and
  Linear Algebra." Then in the same chat, "when is the first one of those on Tuesday?" → "Your first class on Tuesday is
  Linear Algebra, starting at 11 am." The second answer is right only because the agent remembered the first.
- From your testing (Dr. Khan): "what i teach", then "any free slot in monday", then "i want a class on 9-10 am for
  theory of comp" were all understood in one chat. The agent then offered Wednesday 9, 10 or 11 am, which you can now
  pick with a click.
- **Tests:** `tests/nts/test_nts_chat.py`, 10 tests: memory reaches the model, a message while the agent waits is the
  answer, a new message replaces an undecided draft, nobody else can write in your chat, the office sees only what
  reaches it, two tabs are two people, the stuck message, the offer-order hint, answers to pick, and the office inbox
  (approve with a message, a message to an escalated request, still there after a restart). One more test in
  `test_nts_extend.py` covers the lecture-or-practical question. All 163 `nts` tests pass.

### Still to know

- **Memory is short on purpose:** the last 4 messages, 600 characters of each answer. That's enough for "and on Tuesday?"
  and keeps the 8k context safe. Facts that last (a teacher's standing rules) stay in the timetable store, not in the chat.
- The fine-tuned agent was trained on single requests, never conversations. It handles follow-ups well in what we've
  tried, but no eval covers them yet. Next: a small scripted multi-turn eval.
- A request that was waiting, or sent for approval, when the server restarted shows as `interrupted`: its draft lived only
  in memory. Saving drafts would fix this.
- Options are read off the question's words, so an unusual question may get none. The text box always works.
- The office's *Observability* tab still shows every trace. It's the developers' view, not the office's inbox.

---

## Round 2 of testing: where the chat failed, and why (done, 2026-10-31)

You tested two chats as Dr. Khan and found the agent looping, forgetting, and booking the wrong day. Each failure was
traced in the telemetry before fixing it.

| What you saw | Why it happened | What we changed |
|---|---|---|
| "when am I free on Tuesday … I can take an extra class": it asked which time, then looped until "stuck" | **Step 1's tool picker** read the message as a *question* (confidence 0.81) and offered only the read tools, so the agent had no `add_class` to call | The picker never narrows when the message asks for a change ("extra", "move", "cancel", "book", "take", "want" …). It also reads the **whole conversation**, so "2 pm" after "I want an extra class" still gets every tool |
| "what i teach" was ignored: it carried on with the extra class | The memory of the last message was its "could you give more detail …" reply, so the model read the new message as that detail | When a chat has history, the agent is told: *answer the latest message; the earlier ones are only context* |
| "what classes told u to schedule" → "I did not schedule any"; "I will not take that class" → "there is no change 1" | Memory said `[Status: interrupted]` without saying whether anything was in the timetable. And each message starts with an empty draft, so change 1 from an earlier message isn't in it | Every remembered message now says what really happened: **"The office approved it: it is in the timetable now"**, "Withdrawn by a later message: not in the timetable", "Never finished: nothing was changed", plus anything the office wrote. The agent is told its draft starts empty |
| It said 2 pm was free, then the class at 2 pm failed | `free_hours` checks only the *teacher*; the section or the room was busy | When `add_class` can't use a time, it now lists the times that work for that class (teacher, section and room all free), **the day you asked for first**, and tells the agent to ask you which one |
| Asked for Tuesday, got Wednesday without being asked, and the office approved it | Nothing stopped a day other than the one in the request | `add_class` warns the agent: *"the request asks for Tuesday, not Wednesday: ask the requester before you submit"* |
| "Another time on Tuesday, or a different day?" came with Yes / No buttons | The option reader didn't recognise long "A, or B?" questions | It now offers the two sides: **Another time on Tuesday** / **A different day** |
| "It was approved, but no notification" | The approval went through the server still running the old code | **A notifications bell** in the header, with a count of unread items: your request approved, not approved (with the office's note) or not committed; a message from the office; another teacher's approved change that moved or cancelled one of your classes. A **pop-up** appears when a new one arrives. Clicking a notice opens its chat. Notices are rebuilt from the log after a restart |

### Tests

- `tests/nts/test_nts_choices.py` (8): the answers read off each kind of question.
- `test_nts_pick.py` (+3): a question that asks for a change gets every tool; the whole conversation decides the tools.
- `test_nts_extend.py` (+2): a taken time lists that day's free times first; another day than asked says to ask first.
- `test_nts_chat.py` (+2): the teacher is told what the office decided (and still is after a restart); a follow-up is
  told to answer the latest message, and what is in the timetable.
- All 178 `nts` tests pass.

### Still to know

- The fine-tuned 4B model still makes mistakes these fixes can only steer: asking the same thing twice, or making an
  offer before checking the draft. The tools now tell it the right next step each time. Training it on
  multi-message chats is the real fix.
- A draft still lives in memory: a server restart while a change waits for approval loses it (`interrupted`).
