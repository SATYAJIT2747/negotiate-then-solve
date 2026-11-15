# Human study: what the raters said (2026-11-15)

> **Later the same day:** after 08:55 UTC, something automated filled the remaining items for every team
> code and for P-01, at up to 67 labels in 12 seconds. `evaluation.study report` now counts only paced
> labels: a label must be at least 3 s from its neighbours in the same code and task (`paced`;
> `--all-labels` counts everything). Human labels left: claims T-1 67 and T-3 66 (33 labelled by both);
> ratings T-3 73 and T-2 42; rules T-1 17; replies T-1 3. Every comment below is from T-3 before 08:55,
> so all of it is human. Human-only results: claims T-1 vs T-3 kappa 0.39; rules T-1 vs judge kappa 0.73
> (n=17). Grounded vs free, team pairs (n=16): clarity 3.56 vs 3.56; acceptability 3.19 vs 3.63
> (Wilcoxon p=0.025, free text rated higher).

Source: `runs/study/labels.jsonl` (main checkout), the last label each rater gave each item.
Raters: T-2 and T-3 rated messages (T-2 left no comments). T-1 and T-3 labelled claims.
All 73 free-text comments are from T-3 (46 on grounded `ours-llm`, 27 on free-text `A2`).
Raters were blind to which configuration wrote each message. They are team members, not faculty,
so read this as expert pilot feedback, not a user study.

## Scores (1-5)

| | n | clarity | acceptability |
|---|---|---|---|
| A2 (free text) | 47 | 3.68 | 3.70 |
| ours-llm (grounded) | 76 | 3.49 | 3.34 |
| T-2 only: A2 / ours | 20 / 30 | 3.95 / 3.77 | 4.00 / 4.00 |
| T-3 only: A2 / ours | 27 / 46 | 3.48 / 3.30 | 3.48 / 2.91 |

By scenario type (ours-llm): contention 3.72/3.53, deadlock 3.80/3.35, policy 3.00/3.00,
squeeze **2.50**/3.00. Squeeze and policy messages are the weakest for both systems.

People liked how A2 read (friendlier, more natural), but A2 is also where the trust problems
were: it leaked private reasons and its summaries didn't match the option cards (see below).
This fits the earlier H3 result that free text leaks private reasons.

## Themes (number of comments: ours-llm of 46 / A2 of 27)

**What raters liked**
- A fairness reason, especially one that gives **both** counts ("Dr. Pillai twice, you once").
  S-010/1 was called "feels fair", and T-3 said "every message should do this". (ours 10)
- A2's friendly tone, greeting, and offer to talk it over. (A2 22)
- Options that keep the same time across three days, or keep a slot on the original day.
- Naming the other person, what the clash is over, and why both can't happen. A2 S-023/3 was
  called "the clearest version so far".

**What raters asked for, most common first**
1. **Why is it me who moves?** (ours 25, A2 21). This is the biggest gap. Most messages never
   say why this owner was asked and not the other. A fairness claim with no tie-breaker made
   things worse: in S-023/2, "both conceded 3 times" left T-3 asking "so why me?"
2. **"Option A keeps your practical on Monday" when it actually moves it** (about 24 / 13).
   This one wrong word made the rater doubt everything else in the message. It comes from the
   distilled explainer data (`data/distill/explain.jsonl`), not from a template.
3. **A reply deadline** (23 / 12), and **a greeting** in grounded messages (21 / 0).
4. **Say which practical is mine.** "The gpu and routers you need for X and Y" reads as if
   both classes are yours. The option cards cut off course names, so raters had to guess. (14 / 7)
5. **The clash isn't explained** (15 / 6). Worst in squeeze messages: the listed reasons
   (colleague busy 9-1, lunch at 1) don't rule out the afternoon that was asked for, and there
   are unexplained third parties ("Who is Dr. Mehta?").
6. **Options that contradict the message.** It says Lab 1 is the only room with GPUs, then
   offers Lab 2. It says someone is busy 9-1, then offers 9 am. S-035/4 explains one class but
   moves a different one.
7. **Choices that are really the same.** B and C are an hour apart, or policy options are the
   same slots with the lecture numbers shuffled, so "three options" is really two. (8 / 4)
8. Smaller points: drop internal codes such as `[P-LUNCH]` and C2-L1 (9 / 2); write "has moved
   their session" instead of "conceded" (5); give the recipient's own count as well (5);
   confirm students are free (4 / 1); refer to options by letter (A2 6); show policy options as
   a short list ("Mon 9, 10, 12 + Tue 9").

**Problems only in A2 (free text)**
- **Privacy leaks**: a colleague's medical appointment, "family care commitments", "clinical
  duties". The rater said: "Please don't share colleagues' medical appointments."
- Summaries that don't match the cards, garbled sentences, vague wording ("shared lab
  resources"), and slips like a "GPU lab" for Computer Networks.

## The point raised outside the survey: is one clash sent in both directions?

Concern: Singh/Hegde (S-023), Menon/Kumar (S-004) and Kapoor/Das (S-022) were each asked to
move for the same clash. If both people accepted, both would move and the slot would be empty.

**Checked: this can't happen. The messages are sent one after another, not together.**
In `runs/negotiation-study-source.json`, each of these runs has 4 rounds. Owner X is asked
twice and declines both times, then owner Y is asked twice and declines too. The run escalates
with "deadlock: every owner asked declined". `NegotiationEngine._resolve`
(`src/agents/negotiation.py`) asks one owner per round. An owner who declines is added to
`keep`. The loop stops the moment the solver becomes feasible, so the first acceptance ends
the negotiation and nobody else is asked.

The rating form shows each message on its own, so raters couldn't see the earlier refusals.
For the report, this means a message sent after a refusal should say so ("Dr. Singh can't
move, so we're asking you"). That also answers theme 1. The rater's other point stands: the
earlier Kapoor options (two on Tuesday morning) were better for the recipient than the later
Wednesday/Thursday 3 pm ones, because staying on the same day disrupts less.

## Changes to consider (not made)

1. Give a "why you" sentence in every message: an earlier refusal, the ledger with both counts,
   or the actual tie-breaker. Don't use the word "fairness" without a reason behind it.
2. Fix "keeps" vs "moves" in the explainer data, or after generation check the day against the
   original placement.
3. Template additions: greeting, a reply deadline, "your <course>" naming, full course names on
   the cards, no internal codes.
4. Squeeze/policy: say plainly what blocks the requested slot, and drop options that break the
   stated rule.
5. Remove near-duplicate options (same day, an hour apart, or the same slots reshuffled).
