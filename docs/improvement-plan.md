# Improving the intents reader and the agent, separately and systematically

2026-11-06. Built on the head-to-head run (`runs/head_to_head/v1/report.md`: **intents 35/41, agent 28/41** after the
gold fixes below) and on the training and evaluation records: the leaderboard of intents suite runs
(`negotiate-then-solve-simplify/runs/leaderboard.jsonl`, 39 runs), the intents recipes and training split, the agent
eval files and exports (`negotiate-then-solve-agentic/runs/eval`, `data/agent-v*`), and the **Kaggle training logs of
both accounts** (logs and small JSON only, no weights: `LMA Major Project/kaggle-logs/{shubhadeepmandal,seronic2001}/`,
about 4 MB: every `result.json`, `result-dpo.json`, `trainer_state.json`, `manifest.json` and notebook log).

The two systems stay separate: each gets its own fixes, data and training runs; the head-to-head set is the shared
test both are scored on.

**Correction to the first version of this plan.** It said the pipeline's policy model made up "§3.1 Examination
weeks" and a week-7 rule. Both are real: `data/policies/exam-regulations.pdf` (week 8 is mid-semester examinations
with no regular classes; changes in weeks 7, 8, 15 and 16 need the Controller of Examinations and are forwarded). The
refusal of the week-8 swap was right; the week-7 move should have been forwarded under §3.2, and the model cited §3.5
(extra classes) and refused instead. Eleven head-to-head queries had used weeks 7 or 8 by accident: they were moved to
other weeks and rerun, and three exam-week queries were added as their own category.

---

## 1. What the records show

### 1.1 Newer versions lost what older ones could do

**Intents reader.** On suites whose content is identical across runs (same fingerprint), outcome match:

| suite (n) | v6 | v7 | v8 e2 / e3 | v11 e2 / e3 | v12 e2 / e3 |
|---|---|---|---|---|---|
| test, frozen held-out (417) | **1.00** | 0.99 | 0.96 / 0.96 | 0.95 / 0.93 | 0.98 / 0.96 |
| test-typo (417) | 0.99 | 0.98 | 0.94 / 0.95 | 0.94 / 0.91 | 0.98 / 0.96 |
| only_times (40) | 0.25 | 1.00 | 0.75 / 1.00 | 1.00 / **0.47** | 0.50 / 1.00 |
| moves (40) | 0.82 | 0.70 | 0.72 / 1.00 | 0.60 / **0.57** | 0.90 / 0.97 |
| exceptions (40) | 0.00 | 1.00 | 0.50 / 0.93 | 0.97 / 0.70 | **0.50** / 1.00 |
| difficulty (40) | 1.00 | 1.00 | 1.00 / **0.82** | 1.00 / 0.90 | 1.00 / 1.00 |
| terms (40) | 0.97 | 1.00 | 0.90 / **0.78** | 1.00 / 1.00 | 1.00 / 1.00 |

(e2 / e3: the adapter after epoch 2 and 3 of 3, the `read-067` / `read-100` files.) A skill learned in one version is
often lost in the next and relearned after; the frozen test fell from 1.00 (v6) to 0.93 (v11). Causes on record:

1. **v7 was an update of v6** (manifest: parent nts-intents-v6, 335 new rows, 1,005 read). The v8 recipe records what
   it lost and why. Since v8 every version trains from the base; that fixed this cause only.
2. **Each version dilutes the older skills.** Every generator adds a fixed 90 rows, and versions add generators (v8:
   14, v11: 18, v12: 22; 3,208 → 4,062 → 4,363 rows). In v12 the corpus requests the frozen test is drawn from are
   1,087 rows (25%; 33% with their 343 noisy twins); generator rows and their twins are 2,379 (55%); each new skill is
   about 100 rows (2.5%).
3. **Nothing during training can see a regression.** The validation split is drawn from the same generators and
   templates as training, and its loss is near zero at every epoch of every version (0.003–0.009, even for v7, the
   version that regressed). The only check inside the notebook is 15 examples ("compile n=15 match 93–100%").
4. **A version is judged mostly by its new suites**; no rule says a release may not lose an older one.

**Agent.** Each version was scored mainly on its own new families:

| eval | v3 e1 / e2 / dpo | v4 e1 / e2 | v6 e1 / e2 / dpo |
|---|---|---|---|
| the 19 families of v3–v4, 2 each | 38 / 37 / 38 of 38 | 37 / 38 | not run as a file |
| v6's 12 new families, 2 each | – | – | 22 / 21 / 22 of 24 |
| the same, bigger department | – | – | 15.3 (3 tries) / 18 / 20 of 24 |

- v6 was never scored on the older families in a saved run. The one full run on record (42 families, 78/84,
  `docs/agentic-improvements.md`) failed `clarify` and `no_reply`, which v3 and v4 always passed.
- The old-family eval is **saturated** (37–38 of 38) and **too small**: the same v6 e1 adapter scored 16, 13 and 17 of
  24 on three tries. Under about 4 of 24 is noise.
- About 30% of v4 and v6 episodes are the agent's own earlier runs (229 of 772 in v6).

### 1.2 Epochs: what the Kaggle logs say

Validation loss at each saved epoch (the `validation` field of each `result.json`):

| run | rows / episodes | epochs | validation loss by epoch |
|---|---|---|---|
| agent v1 | 190 ep | 3 | 0.122 → 0.105 → 0.103 |
| agent v2 | 160 ep | 3 | 0.182 → 0.175 → **0.177** |
| agent v3 | 458 ep | 2 | 0.1372 → **0.1377** |
| agent v4 | 485 ep | 2 | 0.121 → 0.115 |
| agent v6 | 772 ep | 2 | 0.138 → 0.134 |
| intents v7 (update of v6) | 1,005 | 3 | 0.0165 → 0.0037 → 0.0031 |
| intents v8 | 3,208 | 3 | 0.0091 → 0.0039 → **0.0055** |
| intents v11 | 4,062 | 3 | 0.0055 → 0.0042 → 0.0029 |
| intents v12 | 4,363 | 3 | 0.0045 → **0.0055** → 0.0026 |
| policy/reply/explain (nts-read100) | 3,009 | 2 | 0.0163 → **0.0168** |

**Agent.** The second epoch helps a little or not at all (−14% for v1 with 190 episodes, then −4%, −3%, +0.4% as the
data grew); the **third never helped** (v1 −2%, v2 +1%). Downstream, epoch 1 and 2 are within noise on every suite.
Training loss at the end of epoch 2 (≈0.12) is close to validation loss (0.115–0.138), so it is not badly
over-fitting; the learned tokens are only 3–5% of each episode, and the second pass mostly sharpens formats it already
has. Verdict: **at most 2 epochs; more episodes buy more than more passes.**

**Intents.** Validation loss cannot decide this (§1.1, cause 3). On the held-out suites, the last epoch is what teaches
the newest, smallest skills (v12: relation_pairs 0.07 → 1.00, limits 0.50 → 1.00, compound 0.60 → 0.97), and the
earlier one is as good or better on the common requests (v12 test 0.98 vs 0.96; v11 exceptions 0.97 vs 0.70,
only_times 1.00 vs 0.47). The extra epoch is buying passes for 2.5%-of-the-data skills while over-fitting the rest.
Verdict: **upsample the small skills and pick the checkpoint by its worst suite, rather than a third epoch for all.**

**A confound in every curve point.** With `CURVE` on (both notebooks, every version since), the learning rate is
`constant_with_warmup` at 2e-4 and the data is read in a fixed stratified order (`SequentialSampler`). No adapter in use
was ever annealed, and "epoch 1 vs epoch 2" is one run stopped at two points at full rate, not a one-epoch model vs a
two-epoch model. The epoch question has not been tested with a decaying schedule.

### 1.3 What else the logs show

- **DPO barely moved the agent.** v6: 343 pairs from one source (`outcome`), 1 epoch at 2e-5, β 0.1. Loss
  0.693 → 0.635, held-out preference accuracy **0.65 on 48 pairs**, and the share of training pairs with a positive
  margin fell (0.63 → 0.58). v4: 83 pairs, accuracy 0.67 on 12, margins fell (0.70 → 0.60). v3: 111 pairs, 0.79 on
  14. The DPO adapters' eval gains (v6: 20 vs 18 of 24 on the big department) are within noise.
- **The policy adapter (nts-read100) is one multitask run, never retrained**: compile 1,062, reply 1,007, policy 523
  (17%), explain 417. Its policy rows are 40% "forbidden", and 212 of their 226 citations are two rules (P-LUNCH,
  P-MAXCONSEC). It has no rows for the exam regulations' "forward to the examination cell" (§3.2); what it does on
  week 7 (refuse, citing §3.5) is the habit the data taught: forbidden, one cited rule.
- **The agent's episodes are longer than its serving context.** v6: mean 6,991 tokens, max 8,585; the server gives
  each request 8,192. The longest training episodes cannot happen at serving time, where old tool results are trimmed.

### 1.4 Gaps the head-to-head found

| failure | system | cause | kind of fix |
|---|---|---|---|
| swaps with a colleague (1/5) | agent | `swap_classes` swaps two classes of one teacher only; all 19 swap episodes are same-teacher | tool + data |
| exam weeks ignored (0/3) | agent | it reads `data/handbook.md` only, never `data/policies` (the exam regulations) | code |
| a week-7 change refused, not forwarded | pipeline (policy adapter) | no training rows for §3.2's forward; 40% "forbidden" rows | data |
| "move … to the afternoon" became Tue/Wed/Thu | intents | 0 training rows of a move to a time of day with the day implied | data |
| a change plus a question: the question became a "clash report" | intents | 0 training rows with a change and a question in one message | data |
| a clarifying answer read as a new wish | intents | `World.clarify` appends the answer and reads all of it again | code + data |
| "before noon" written as 12–5 pm; "marked you away" after retracting it | agent | the final message is free text, not the draft | code + DPO pairs |
| loops after repeated refusals | agent | the repeat guard blocks exact repeats only | code |

(Moved to week 6, "Room 101 is broken" and another teacher's class both passed for the intents reader; in week 7 both
had failed. Whether the exam-week rule or run-to-run variation made the difference was not separated; the training
data still has no row of "my class needs another room" (all 39 `room_closed` rows are facilities notices).)

---

## 2. The plan

Four phases, each ending with the same measurement (Phase 0).

### Phase 0: measure before changing anything (≈1 day, no training)

1. **One regression matrix per system, run on every candidate checkpoint.**
   - Intents: the frozen test, test-noisy, test-typo and every suite (`evaluation.suites`), per suite against the best
     earlier version.
   - Agent: all 42 families at **5 per family** (not 2), both departments, 2 seeds.
   - Both: `evaluation.head_to_head` (41 queries now; §3 grows it).
2. **Measure the noise floor once** (each current adapter 3× on the same suites).
3. **A release rule**: no suite below its previous best minus the noise band; choose versions by their worst suite.
4. **In the notebooks**: replace the 15-example check with a few held-out suites at each curve point (they are small:
   40 rows each), so a regression shows up during training, not after.
5. **Keep the logs**: copy `result*.json` and `manifest.json` into the repo after each run (the commands are in the
   memory note; a few KB each).

### Phase 1: fix what training cannot (≈2–3 days, no training)

Intents pipeline:
1. Check each policy verdict against the retrieved text (the cited section must exist and support it), and add the
   exam regulations' "forward" outcome: §3.2 is a forward to the examination cell, not a denial.
2. Permission in the shared core: a sender may only constrain their own classes and time, whoever read the request.
3. A clarifying answer fills what was missing: keep the first reading and give the reader the question and the answer.

Agent:
4. Load `data/policies` into the agent's handbook (as the pipeline does), so the exam weeks reach it.
5. `swap_with(colleague, ...)`: both moves in the draft and a consent ask to the colleague.
6. The final message written from the draft and the solver's diff.
7. After K refusals of one kind, only the end tools remain.

### Phase 2: the data (≈3–4 days)

**Intents v13** (from the base):
1. **Gap generators** (§1.4): a move to a time of day (same day), a change plus a question, the answer to a clarifying
   question (request + question + answer → the full reading), another teacher's class (refuse or forward); with
   held-out suites.
2. **Keep each skill's share**: replace "90 rows per generator" with a target mix: corpus requests (with their twins)
   at a floor of about 40% (33% now, and falling each version), generators sampled by size^0.5, a floor of about 2% per
   kind.
3. **Upsample the newest and weakest kinds** (×2–3, chosen by suite scores) instead of a third epoch.
4. **More mined rows** (the model's mistakes on suites, the head-to-head and demo logs; 120 now), and **fewer
   near-duplicates** (a cap per template).

**Policy adapter** (it was trained once and never revisited):
5. Rebuild its policy rows: verdicts balanced against the real distribution (most requests are allowed), citations
   spread over the handbook **and** `data/policies` (exam weeks, guest faculty, lab notice, bookings), a "forward"
   (needs the examination cell or the office) outcome, and rows where the retrieved text does not apply (the model
   must not cite it). Train it on its own (policy, reply, explain), not with compile, now that compile has its own
   adapter.

**Agent v7** (from the base):
6. New families: swaps with a colleague (agreed, refused), absences for part of a day, a change plus a question, exam
   weeks (forward), two-message conversations.
7. A floor of 25 episodes per family; at most 20% of episodes from the agent's own runs, and only those that pass the
   stricter checks.
8. **DPO that can move the model**: more pairs (≥ 800) from more sources (`outcome` and `fix`, plus pairs built from
   the §1.4 failures: a reply that claims more than the draft vs one written from it; a loop vs ending with the right
   tool), β and the rate tuned on a held-out pair set large enough to read (≥ 150 pairs), and kept only if the margins
   rise.
9. Episodes that fit the serving context (≤ 8,192 tokens, or serve with a longer context).

### Phase 3: the training recipe (controlled runs; intents first, they are short)

| run | question | compare |
|---|---|---|
| R1 | Does annealing help? | constant 2e-4 (now) vs cosine to 0, same steps |
| R2 | How many epochs, with a normal schedule? | cosine, 1 / 2 / 3 epochs, separate runs |
| R3 | Upsampling vs epochs | 2 epochs + upsampled weak kinds vs 3 epochs plain |
| R4 | Learning rate | 2e-4 vs 1e-4 (less drift from the base, less forgetting) |
| R5 | Capacity | LoRA rank 16 (now) vs 32 |
| R6 | Averaging | the e2 and e3 adapters averaged (no training) |

To keep learning-curve points without the confound: warmup–stable–decay (constant rate, and at each curve point a
short decay branch, about 10% of the steps, saved as that point's adapter).

For the agent, the logs already answer most of R2 (≤ 2 epochs); run R1 (cosine vs constant, 2 epochs) and the DPO
changes of item 8, then the intents settings for rate and rank if they transfer. Incremental updates stay off (the v7
lesson); if cost forces one, replay at least 3× the new rows, stratified by kind.

### Phase 4: release and report

- Each system's winner passes the release rule with the best worst-suite score.
- Re-run the head-to-head (about 120 queries by then) and both regression matrices.
- For the report: ablations for the data mix, schedule, epochs, rate, rank and DPO, and the before/after head-to-head
  by category.

---

## 3. The head-to-head set

41 queries are enough to see the categories, not to separate two versions of one system. Grow it to about 120 (6–10 per
category), the §1.4 gaps included, written against the demo department's real timetable and rules: weeks 7, 9 and 11
have their own changes, and weeks 7, 8, 15 and 16 fall under the exam regulations, so a change query uses them only on
purpose. Keep it out of every training mix.
