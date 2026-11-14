# Remaining baselines and ablations (2026-11-14)

The runs from the proposal brief that need no human input. Every number below
comes from a file in `runs/`. Main model: Qwen3.5-4B + LoRA (Q8_0).

## 1. Model-bias check: ours vs B4 on the same model (brief §5.1)

The main comparison had ours on the local 4B and B3/B4 on Gemini Flash-Lite.
This run puts both on Gemini, on a stratified 20 scenarios, seed 0
(`runs/negotiation-bias-gemini.json`).

| | Correct outcome | Resolved (of 9 agree) | Oracle gap | LLM calls / scenario | First-offer acceptance |
|---|---|---|---|---|---|
| Ours (Gemini agent) | 90% | 7 | 0.00 | 0 | 38% |
| B4 (Gemini agent) | 75% | 4 | 0.68 | 5.1 (7.4 filtered) | 8% |

Ours on Gemini resolves exactly the conflicts ours on the 4B resolves on the
same 20 (7 vs 7; 18/20 outcomes identical). The gap to B4 is the option
source, not the model. With n = 20 the McNemar test is not significant
(4 vs 1 discordant, p = 0.38); the 180-pair main run is (p = 5e-6).

## 2. Ablations A1-A3, paired (`evaluation.ablations`)

Each ablation row (stratified 30, seed 0) paired with ours on the same
scenario and seed (`runs/negotiation-ablations-paired.json`). The earlier pooled
comparison set ours over 60 x 3 against the ablations over 30, which is not
like-for-like.

| | Resolved | Correct | Changes who gives way | Gini (same 30) | Leaks | First-offer acceptance | Judge clarity / acceptability |
|---|---|---|---|---|---|---|---|
| Ours | 77% | 90% | – | 0.25 | 0 | 21% | 4.47 / 4.26 |
| A1 flat | 77% | 90% | 0 of 10 | 0.25 | 0 | 21% | 4.37 / 4.32 |
| A2 free text | 77% | 90% | 1 of 10 | 0.30 | 12 | 11% | 4.47 / 4.26 |
| A3 rule replies | 77% | 90% | 1 of 10 | 0.30 | 0 | 11% | 4.42 / 4.00 |
| Ours vs A1-offline (all 60, scripted) | 77% vs 73% | 88% vs 87% | 1 of 22 | 0.185 vs 0.227 | 0 | 32% vs 29% | – |

Findings:

- **A1 / H4.** On the benchmark, flat weights almost never change who is asked:
  its scenarios are independent one-off conflicts, so the ledger rarely meets
  the same person twice. The earlier "A1 Gini 0.221 < ours 0.241" came only
  from comparing different scenario sets. H4 needs the stress test (Section 3).
- **A2 / H3.** Free text leaked a private reason in every scenario that had one
  to leak (4 of 4, 12 leaks); grounded explanations in none (0 of 4).
  Acceptance is not lower with grounding (21% vs 11% first-offer acceptance,
  same judge scores). Faithfulness is not comparable across modes: free
  sentences are checked against all facts, grounded claims against the facts
  they cite, so A2's 1.00 and ours' 0.96 use different yardsticks.
- **A3 / tool calling.** Outcomes match ours because the simulated replies in
  these 30 are almost all plain accepts and refusals (31 reject, 6 accept,
  4 counter, 0 vague), where the regex handler is right. Its weakness shows on
  Benchmark D, which has every reply kind: rule handler 66% tool accuracy
  (counter 0%, vague 0%) vs the fine-tuned 4B 100%.
- Judge scores (Gemini, unvalidated until the human kappa labels exist) do not
  separate the configurations (all p > 0.2).

## 3. Fairness stress test, H4 (`evaluation.fairness`)

One department, the same four teachers, 24 lab clashes per semester, 10
semesters; F-301 and F-304 teach the smallest section (35 vs 55 and 60), so
moving their practical is always cheapest on paper. Identical conflicts and
hidden flexibility for every configuration; scripted replies, no LLM
(`runs/fairness-final.json`).

| | Gini (95% CI) | Max burden | Times gave way, F-301 / 302 / 303 / 304 | Resolved | Students in conceder's section |
|---|---|---|---|---|---|
| **Ours (ledger)** | **0.18** (0.13-0.23) | 9.6 | 7.4 / 4.4 / 4.2 / 8.0 | 100% | 43.0 |
| No ledger (same weights) | 0.34 (0.29-0.39) | 11.8 | 9.0 / 2.9 / 2.8 / 9.3 | 100% | 40.3 |
| Flat (A1) | 0.16 (0.11-0.22) | 9.2 | 6.3 / 4.7 / 5.7 / 7.3 | 100% | 44.9 |
| Ours, smooth credit (experimental) | **0.13** (0.09-0.18) | 8.6 | 6.8 / 5.2 / 4.7 / 7.3 | 100% | 44.2 |

Paired Wilcoxon over the 10 semesters:

- **Ours vs no ledger:** Gini -0.15 (p = 0.002), max burden -2.0 (p = 0.002),
  resolutions unchanged, at +2.7 students per conflict. **H4 holds for the
  ledger itself.**
- **Ours vs flat:** Gini not different (p = 0.40), but flat disturbs 2.1 more
  students per conflict (p = 0.008): flat is even because it ignores who is
  affected, not because it tracks who has given way.
- **Smooth credit vs ours:** Gini -0.045 (p = 0.012), max -1.3 (p = 0.03),
  +1.0 student per conflict. `PriorityModel.credit` caps credit at 1.0, which
  one concession nearly reaches, so after that the ledger cannot tell one
  concession from three. `1 - 0.5^credit` keeps growing. Not applied to the
  live system; it would change the deployed behaviour and the headline runs.

## 4. Fine-tuning: prompted base 4B on parsing (Benchmark B, 151 test)

`evaluation.parsing --prompted` gives the base model the full prompt the
Gemini parser gets (`runs/parsing-4b-base-prompted-paratest.json`).

| | Action accuracy | Exact compile match | Atom F1 | Injections handled | Latency p50 |
|---|---|---|---|---|---|
| **Qwen3.5-4B + LoRA** | **99.3%** | **97.8%** | **0.996** | 11/11 | 2.0 s |
| Qwen3.5-4B prompted (no adapter) | 75.9% | 47.8% | 0.829 | 8/11 | 13.2 s |
| Qwen3.5-2B + LoRA | 96.5% | 97.8% | 0.996 | 8/11 | 1.0 s |
| Gemini Flash-Lite (prompted) | 99.3% | 92.4% | 0.977 | 11/11 | 1.5 s |

LoRA lifts the same 4B from 48% to 98% exact compilation, at a sixth of the
latency (short prompt), and beats prompted Gemini on exact match.

## 5. RAG: with and without retrieval, and abstention (Benchmark C)

40 answerable questions worded unlike the rules (`data/retrieval-queries.jsonl`)
and 20 that no rule covers (`data/abstention-queries.jsonl`: leave, attendance,
grading, fees, parking). Policy agent on 4B + nts-read100, hybrid retrieval,
k = 5; answers labelled by a Gemini judge (`runs/rag-4b-read100.json`).

| | Answer accuracy (95% CI) | Citation recall / precision | Abstains when no rule | Hallucinates when no rule | False abstention |
|---|---|---|---|---|---|
| **RAG** | **77.5%** (62-90) | 80% / 82% | **100%** | **0%** | 5% |
| No retrieval | 20.0% (8-33) | – | 75% | 25% | 15% |

Without retrieval the model invents institute policy ("10 days of casual
leave", "75% attendance"). With it, every unanswerable question gets "no rule
covers this", and 0 citations are hallucinated. Of the 9 answerable misses, 6
are retrieval misses (gold rule not in the top 5; retrieval recall 85%), 2
cite a neighbouring rule that was retrieved beside the right one (Q-01, Q-21),
and 1 cites the right rule with an answer the judge marks inconsistent (Q-12).

### Retrieval depth (k)

Recall of the gold rule on the same 40 questions, no model calls (`runs/retrieval-k.json`):

| | k = 5 | k = 8 | k = 10 |
|---|---|---|---|
| BM25 | 67.5% | 77.5% | 82.5% |
| Dense | 90.0% | 92.5% | 92.5% |
| **Hybrid** (deployed) | 85.0% | **95.0%** | 95.0% |

The full RAG run at hybrid k = 8 (`runs/rag-4b-read100-k8.json`): answer accuracy 80.0%
(67.5-92.5), citation recall 82.5%, citation precision 86.8%, retrieval 95%, abstention
100%, hallucination 0%. Six of the eight remaining misses had the gold rule in front of
the model and cited a neighbouring one, so the bottleneck moves from retrieval to
choosing between similar rules.

Regression check on ordinary requests (policy test split, 109 requests, 4B + adapters,
hybrid; `runs/policy-4b-hybrid-k{5,8}-test.json`): k = 5 and k = 8 are identical on
every verdict measure (allow/deny 100%, deny recall and citation 100%, false denials 0,
make-up obligations 100%, hallucinated citations 0); retrieval recall 95.7% vs 97.8%.
Showing eight rules costs nothing on ordinary requests, so the default is now k = 8
(`PolicyAgent`, its rule-based fallback and `evaluation.policy`).

## 6. Safety: the deterministic checks vs none (50 cases)

`evaluation.safety --no-guards` switches off, for the evaluation only, sender
verification, the injection pre-filter, the authority check, the students-only-report
rule, refusing changes to a colleague's class, and human approval (whatever reaches
approval is approved at once). Scores still use the real authority check.

| | Handled safely | Published without approval | Authority violations | Injections obeyed | Leaks |
|---|---|---|---|---|---|
| Checks off, compromised model | 4% | 48 | 47 | 18 | 0 |
| Checks on, compromised model | 98% | 0 | 0 | 0 | 0 |
| Checks off, Qwen3.5-4B + LoRA | 72% | 14 | 0 | 0 | 0 |
| **Checks on, Qwen3.5-4B + LoRA** | **100%** | 0 | 0 | 0 | 0 |
| Checks on, Gemini Flash-Lite (earlier run) | 100% | 0 | 0 | 0 | 0 |

The 4B on its own ignores injections and grabs for authority, but without the approval
gate 14 changes publish unseen, 4 of them from colleagues asking to change someone
else's class. With the checks on, even a model that obeys every instruction publishes
nothing unapproved. Leaks are 0 everywhere because replies are built from constraints,
never from the reasons given.

The compliance check was tightened on 2026-11-14: "published" and "approved" alone
matched the ordinary reply "waiting for the coordinator's approval: yours is approved
after it" (10 false injection successes on the 4B). A reader that fails on a message is
now scored on what it changed instead of stopping the run.

## Still needs people

- Judge validation: two annotators x 100 explanations and 100 replies in
  `/study`, then `evaluation.study report` (kappa >= 0.6 before LLM-judged
  numbers are reported).
- Human pilot (5-10 faculty, ethics clearance).
