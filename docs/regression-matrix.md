# Regression matrix (Phase 0)

From `C:\Users\Shubh\Desktop\LMA Major Project\kaggle-results`.

## Intents: outcome match per suite

| suite | misses/v12-e3 | misses/v13 | v12-e3 | v12-e3/r1 | v13 | v13/r1 | v13b-e1 | v13b-e2 | v13b-e3 |
|---|---|---|---|---|---|---|---|---|---|
| test | 0.959 | 0.981 | 0.964 | 0.957 | 0.983 | 0.981 | 0.950 | 0.964 | 0.966 |
| test-noisy |  |  | 0.954 | 0.952 | 0.981 | 0.983 | 0.923 | 0.945 | 0.950 |
| test-typo |  |  | 0.952 | 0.952 | 0.974 | 0.978 | 0.930 | 0.950 | 0.945 |
| answered |  |  | 0.675 | 0.650 | 1.000 | 1.000 | 0.825 | 0.975 | 0.675 |
| choices |  |  | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| clubs | 0.950 | 0.925 | 0.975 | 0.950 | 0.925 | 0.925 | 0.950 | 1.000 | 0.950 |
| compositional |  |  | 0.857 | 0.857 | 0.929 | 0.929 | 0.714 | 0.714 | 0.714 |
| compound | 0.975 | 0.650 | 1.000 | 0.975 | 0.650 | 0.650 | 0.900 | 0.975 | 1.000 |
| dated |  |  | 1.000 | 0.975 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| difficulty |  |  | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.975 | 1.000 |
| exceptions | 1.000 | 0.950 | 1.000 | 1.000 | 0.925 | 0.950 | 0.650 | 0.975 | 0.700 |
| firmness |  |  | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| limit_firmness |  |  | 1.000 | 0.975 | 1.000 | 1.000 | 1.000 | 1.000 | 0.700 |
| limits |  |  | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.775 |
| move_pairs |  |  | 0.950 | 0.875 | 0.925 | 0.950 | 0.925 | 1.000 | 0.825 |
| moves |  |  | 0.975 | 0.975 | 0.975 | 0.975 | 0.900 | 1.000 | 0.850 |
| multi |  |  | 0.925 | 0.875 | 1.000 | 1.000 | 0.925 | 1.000 | 1.000 |
| only_times |  |  | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| out_for_me | 0.950 | 0.525 | 0.975 | 0.950 | 0.525 | 0.550 | 1.000 | 1.000 | 1.000 |
| periods |  |  | 1.000 | 1.000 | 1.000 | 1.000 | 0.775 | 1.000 | 1.000 |
| reasons |  |  | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| relation_pairs |  |  | 0.875 | 0.850 | 0.900 | 0.900 | 1.000 | 1.000 | 0.650 |
| relations | 1.000 | 0.925 | 1.000 | 1.000 | 0.850 | 0.925 | 1.000 | 1.000 | 0.650 |
| same_day_part | 0.275 | 0.700 | 0.275 | 0.275 | 0.725 | 0.750 | 0.325 | 0.550 | 0.050 |
| terms |  |  | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| vague |  |  | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| week_patterns |  |  | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.825 | 1.000 |

Noise (v12-e3 run 2x): widest spread 0.075; per suite answered 0.025, clubs 0.025, compound 0.025, dated 0.025, limit_firmness 0.025, move_pairs 0.075, multi 0.050, out_for_me 0.025, relation_pairs 0.025, test 0.007, test-noisy 0.002

## Agent: passed / n per family (both departments)

| family | v6-dpo | v6-dpo/pol | v7-dpo | v7-dpo/fix | v7-dpo/pol | v7-sft | v7-sft/fix | v7-sft/pol |
|---|---|---|---|---|---|---|---|---|
| after_lecture | 6/10 |  | 7/10 |  |  | 1/10 |  |  |
| ambiguous | 10/10 |  | 10/10 |  |  | 10/10 |  |  |
| away | 10/10 |  | 10/10 |  |  | 10/10 |  |  |
| away_and_cancel | 9/10 |  | 10/10 |  |  | 10/10 |  |  |
| away_days | 10/10 |  | 10/10 |  |  | 10/10 |  |  |
| away_part | 10/10 |  | 9/10 |  |  | 9/10 |  |  |
| away_pinned | 10/10 |  | 10/10 |  |  | 10/10 |  |  |
| cancel_one | 10/10 |  | 10/10 |  |  | 10/10 |  |  |
| clarify | 6/10 |  | 7/10 |  |  | 6/10 |  |  |
| clarify_negotiate | 10/10 |  | 9/10 |  |  | 10/10 |  |  |
| club_booking | 10/10 |  | 10/10 |  |  | 10/10 |  |  |
| counter_offer | 7/10 |  | 2/10 |  |  | 1/10 |  |  |
| courses_apart | 10/10 |  | 9/10 |  |  | 10/10 |  |  |
| day_cap | 0/10 |  | 0/10 | 10/10 |  | 0/10 | 10/10 |  |
| days_cap | 2/10 |  | 2/10 | 8/10 |  | 2/10 | 7/10 |  |
| even_weeks | 8/10 |  | 9/10 |  |  | 7/10 |  |  |
| exam_week |  | 8/10 | 0/10 |  | 10/10 | 0/10 |  | 9/10 |
| except_week | 7/10 |  | 8/10 | 10/10 |  | 7/10 | 10/10 |  |
| fallback | 8/10 |  | 4/10 |  |  | 6/10 |  |  |
| free_gap | 10/10 |  | 10/10 | 10/10 |  | 10/10 | 10/10 |  |
| hedged_yes | 10/10 |  | 9/10 |  |  | 10/10 |  |  |
| hour_limit | 5/10 |  | 5/10 | 5/10 |  | 5/10 | 10/10 |  |
| impossible | 10/10 |  | 10/10 |  |  | 10/10 |  |  |
| injection | 10/10 |  | 10/10 |  |  | 10/10 |  |  |
| make_up | 10/10 |  | 10/10 |  |  | 10/10 |  |  |
| move | 6/10 |  | 8/10 |  |  | 6/10 |  |  |
| move_apart | 8/10 |  | 4/10 |  |  | 5/10 |  |  |
| negotiate | 10/10 |  | 9/10 |  |  | 10/10 |  |  |
| negotiate_refused | 10/10 |  | 8/10 |  |  | 8/10 |  |  |
| no_reply | 9/10 |  | 1/10 |  |  | 3/10 |  |  |
| not_allowed | 10/10 |  | 10/10 |  |  | 10/10 |  |  |
| or_windows | 5/10 |  | 5/10 |  |  | 4/10 |  |  |
| preference | 10/10 |  | 4/10 |  |  | 8/10 |  |  |
| q_classes | 10/10 |  | 10/10 |  |  | 10/10 |  |  |
| q_colleague_free | 6/10 |  | 10/10 |  |  | 10/10 |  |  |
| q_free | 10/10 |  | 10/10 |  |  | 10/10 |  |  |
| ranked_move | 8/10 |  | 6/10 |  |  | 8/10 |  |  |
| refused_alternative | 7/10 |  | 7/10 |  |  | 10/10 |  |  |
| room_closed | 10/10 |  | 10/10 |  |  | 10/10 |  |  |
| rules_question | 8/10 |  | 5/10 |  |  | 6/10 |  |  |
| swap | 9/10 |  | 9/10 |  |  | 9/10 |  |  |
| swap_colleague | 9/10 |  | 9/10 |  |  | 9/10 |  |  |
| swap_colleague_refused | 10/10 |  | 10/10 |  |  | 10/10 |  |  |
| three_changes | 9/10 |  | 8/10 |  |  | 8/10 |  |  |
| two_changes | 10/10 |  | 10/10 |  |  | 10/10 |  |  |
| two_windows_off | 10/10 |  | 9/10 |  |  | 9/10 |  |  |
| week_range | 10/10 |  | 10/10 |  |  | 10/10 |  |  |
| weekend | 10/10 |  | 10/10 |  |  | 10/10 |  |  |
| window_move | 10/10 |  | 9/10 |  |  | 10/10 |  |  |
| **all** | **412/480** [0.82, 0.89] | **8/10** [0.49, 0.94] | **381/490** [0.74, 0.81] | **43/50** [0.74, 0.93] | **10/10** [0.72, 1.00] | **387/490** [0.75, 0.82] | **47/50** [0.84, 0.98] | **9/10** [0.60, 0.98] |

Noise (v6-dpo vs v6-dpo/pol, the same job twice): 420 episodes differ over 49 families; the widest family differs by 10.

Noise (v7-dpo vs v7-dpo/fix, the same job twice): 374 episodes differ over 49 families; the widest family differs by 10.

Noise (v7-dpo vs v7-dpo/pol, the same job twice): 391 episodes differ over 49 families; the widest family differs by 10.

Noise (v7-dpo/fix vs v7-dpo/pol, the same job twice): 53 episodes differ over 49 families; the widest family differs by 10.

Noise (v7-sft vs v7-sft/fix, the same job twice): 386 episodes differ over 49 families; the widest family differs by 10.

Noise (v7-sft vs v7-sft/pol, the same job twice): 396 episodes differ over 49 families; the widest family differs by 10.

Noise (v7-sft/fix vs v7-sft/pol, the same job twice): 56 episodes differ over 49 families; the widest family differs by 10.
