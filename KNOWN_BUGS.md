# Known bugs

Bugs seen in the app, with the cause and the fix proposed. Newest first. The adapter's scores and its misses
on the test set are in [PROGRESS.md](PROGRESS.md); this file is for what goes wrong in use.

**Status:** open = not fixed yet; fixed = fixed on the branch named.

## Timetable questions (`f-lookup`, chat → `/api/ask`)

Seen on **Saturday 24 October 2026** (week 13 ended the day before), asking as Dr. Menon, with the app on the
local model (`nts-lookup-v1`).

### B6. One dropped connection turns a question into a request (open)

| You ask | You get | Should get |
|---|---|---|
| what courses do dr.iyer take ? on monday and tuesday | request **R-00036**, answered by the policy check with nonsense (B7) | *Dr. Iyer teaches in week 13: Compiler Design practical, Monday 2–4 pm (Lab 3); …*, which is what the app answers when the server is up |

**What happened** (the app's event log; local time):

| Time | Event |
|---|---|
| 11:39:32.143 | `model_unreachable: ConnectionResetError [WinError 10054] An existing connection was forcibly closed by the remote host` |
| 11:39:32.206 | R-00036 created |
| 11:40:04 | `model_reachable` again |

1. `/api/ask` sent the question to the adapter; the server reset the connection during the call (it also did at
   11:34 the same day, and on 23 October).
2. `World.read_question` fell back to the regex, as designed, and marked the model down for 30 s.
3. The regex can't read it: it has no pattern for "what courses does X take" (it knows "teach", "classes",
   "schedule", "free", …), so it said "not a timetable question". (The name is found either way; the tidy "what
   courses does Dr. Iyer take on monday and tuesday?" fails the same way.)
4. The chat (`ConversationPane.send`) then sent the message as a request, without a word.

**Fix proposed.**
1. A connection reset is usually momentary: try the model **once more right away** before falling back.
2. The regex: "courses … take / handle / teach" as a `describe` / `classes_of` question.
3. To discuss: when the model is unreachable and the regex can't read a message that looks like a question
   (ends with "?", starts with what / when / who / is / any …), say *"I can't answer questions right now; try
   again in a minute, or send it as a request"* instead of turning it into a request silently.

### B7. The policy check answers a non-policy question with its own instructions (open; the team's adapter)

| Message (sent as a request, see B6) | Reply |
|---|---|
| what courses do dr.iyer take ? on monday and tuesday | *This is a policy question about course scheduling, which is answered by citing the relevant rules and setting the verdict to allowed.* |

**Cause.** The request pipeline classed it as a policy question (`answer`), and the policy adapter
(`nts-read100`) was asked to answer it from the rules. No rule is about who teaches what, and the adapter wrote
a sentence about its task instead of an answer (no rule cited, verdict "allowed"). The reply is shown as is.

**Fix proposed.** In the policy check: an answer that cites no rule should not be shown as an answer; reply
*"I couldn't find a rule about that"* (the relevance cut-off already in the RAG roadmap), or hand a question
about the timetable back to the lookup. Worth an issue for the team, since it's their adapter and pipeline.

### B1. "today" on a weekend: "I don't know Sat" (open)

| You ask | You get | Should get |
|---|---|---|
| any free slots available today ?? | *I don't know Sat. Could you check the name?* | *Today is Saturday: there are no classes. Ask about Monday or next week.* |
| what do I teach today? | last Monday's classes (week 13), without saying so | the same as above |

**Cause.**
- The adapter writes the day as `Sat`, either in `days` (rejected: days are Mon–Fri) or in `unknown` (the field
  for names the department doesn't have), which gives "I don't know Sat".
- It never learned this: the training data has no "today" on a weekend (the generator skips "today" templates
  when today is a Saturday or Sunday).
- The regex fallback quietly answers for next Monday instead, without saying it changed the day.

**Fix proposed.** In code, before either reader: a question about *today / this morning / this afternoon /
tonight* on a Saturday or Sunday gets the fixed reply above. And when the model still answers with `Sat` or `Sun`,
the reply *"There are no classes on Saturday."* instead of a rejected answer. In `nts-lookup-v2`: weekend "today"
questions in the training data.

### B2. Nobody named: "I don't know any" / "I don't know some" (open)

| You ask | The adapter's tries | You get |
|---|---|---|
| any free slots available next monday ?? | 1: `free_hours`, Mon, week 14, **nobody named** → rejected; 2: `free_rooms` → rejected; 3: `free_rooms`, `unknown: ["any"]` | *I don't know any. Could you check the name?* |
| what are some free slots on monday ?? | the same pattern | *I don't know some. Could you check the name?* |
| free slots next monday? | `free_hours`, Mon, nobody named, three times | *I couldn't work out that question.* |

Should get: your free hours on Monday (week 14).

**Cause.**
- The question names nobody. The adapter learned "free slots" only together with *my* / *I* ("my free slots",
  "am I free"), so it leaves the person out.
- The regex has a rule for this (nobody named = the person asking); `ToolAgent`'s checks don't, so they reject
  the answer: *"free_hours needs at least one teacher, section or room"*.
- The retry message then pushes the model to change something; it ends up putting an ordinary word of the
  question ("any", "some") into `unknown`, which is trusted, and the reply is "I don't know any".

**Fix proposed.**
1. In `ToolAgent`'s checks: `free_hours` / `classes_of` with no teacher, section, room or course → the person
   asking (a teacher, or a class representative's section), as the regex does.
2. `unknown` and `ambiguous` only count for something that can be a name: day names, dates and ordinary words
   ("any", "some", "all", "free", "slots", "today", …) are ignored there.
3. In `nts-lookup-v2`: "free slots" questions that name nobody.

### B3. A misspelt day is dropped: "free slots on monmday" answers the whole week (open)

| You ask | You get | Should get |
|---|---|---|
| free slots on monmday | your free hours **Monday to Friday of week 13** | your free hours on Monday (of week 14: see B4) |

**Cause.** "monmday" is not a day name, so the day is left out and the answer covers the whole week without a
word about it. Both readers would do this (the reply doesn't show which one answered): the adapter never saw
misspelt days, and the regex reads `mon`, `monday`, `mon.`, not misspellings.

**Fix proposed.** Read days with a little tolerance for typos (one letter added, missing or swapped:
"monmday", "tuseday", "wendesday", "thrusday") before the question goes to either reader; the request reader
already does this kind of spelling repair (`language.rule_parser.normalise`: "thrusday", "aftrnoon"). And when
a reply covers the whole week because no day was understood, say so.

### B4. "This week" on a weekend is the week that just ended (open)

| You ask (on Saturday 24 Oct) | You get | Expected by most people |
|---|---|---|
| free slots on monmday, my free slots, … | **week 13** (19–23 Oct, already over) | week 14 (26–30 Oct), the week the Timetable page already shows on a weekend |
| any free slots available next monday ?? | week 14 | week 14 |

**Cause.** The current week is counted from the semester start (`World.current_week()`), so Saturday and
Sunday still belong to the week that has just ended. The Timetable page, by contrast, shows the coming week on a
weekend, so the chat's answers and the page disagree.

**Fix proposed.** On a Saturday or Sunday, questions take the coming week as "this week" (as the Timetable page
does), and the reply names the week, as it already does ("(week 14)"). To decide: is "next week" on a Saturday
then week 15? (Probably: "next week" said on a weekend usually means the coming one; worth asking a few teachers.)

### B5. The adapter's misses on the test set (open, for `nts-lookup-v2`)

34 of 362 test questions, all listed in PROGRESS.md. The ones a user will meet most:

| You ask | You get | Should get |
|---|---|---|
| are there any labs free on Mon? | free labs **for the whole week** (the day left out) | free labs on Monday |
| can I have Thursday free? | Thursday's free hours, *with the request words first, as in the app: a request* ✓ | a request |
| could Rao take my class on Friday? | Rao's free hours on Friday (not caught by the request words) | a request |
| our AI class clashes with another class on Monday | the changes this week | a request (a clash report) |
| any labs free on Aug 19? | *I don't know Aug 19* | free labs on Wednesday 19 August (week 4) |
| what equipment does Lab-3 have? | *I couldn't work out that question* | Lab 3's equipment |

**Fix proposed.** Training data for these patterns in `nts-lookup-v2`, always with wordings different from the
test set. "Could X take my class" and "clashes" can also go into the request words, which run first.

## Elsewhere in the app (not this branch)

Found while testing; they belong on their own branches from `simplify` or as GitHub issues.

| Bug | Where | Status |
|---|---|---|
| "move my Friday class to Tuesday" read as "Tuesday **or** Friday" by the fast path, so nothing moves ("already fits") or the wrong meetings are targeted | `language/rule_parser.py` (fast path) | open: issue drafted |
| An answer to a clarifying question is read by the fast path, not by the model that asked it | `pipeline/orchestrator.py` `_parse` | open: issue drafted |
| The answer to a clarifying question is shown glued to the original message | the request chat (frontend) | open: issue drafted |
| System One's threshold is tuned on the classifier's labels, not on whether the regex reads correctly: 99% "right" vs 87% actually read right on the test split | `language/system_one.py` `tune_tau` | open |
| `test_semester_api` fails from 16 October 2026 on: it books a club room on a fixed date (15 October 2026), now in the past | `tests/test_semester.py` | open: issue drafted |
| Requests stay in "received" after an error (e.g. the server resetting the connection) | the request pipeline | open: issue drafted |
| The reader drops dates, times and attendees from room bookings ("book lab 2 on 27 oct 12 to 1 pm") | `nts-intents-v6-read100` | open |
