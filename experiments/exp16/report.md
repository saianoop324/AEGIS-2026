# Experiment 16: extra metrics for the scorecard

Computed from existing result files (Qwen: exp14 / exp14fix; Nemotron: nemotron_deberta_*; held-out classifiers: exp14_heldout_*), DeBERTa write gate. No new model runs.

## 1. Confidence intervals and significance

Attack success rate (ASR) = harmful tool calls executed / attack runs, hand-written BankDesk sets (attack + in-gate), 3 seeds. Wilson 95% interval in brackets. Reduction = 1 - ASR/ASR(unprotected), with a 95% bootstrap interval over scenarios. p = exact McNemar test on paired (scenario, seed) runs.

### Qwen3.5-9B, BankDesk

| arm | ASR [95% CI] | reduction [95% CI] | vs unprotected (p) | vs NeMo Guardrails (p) |
|---|---|---|---|---|
| unprotected | 54/78 = 0.69 [0.58, 0.78] | - | - | - |
| nemo_rails | 42/78 = 0.54 [0.43, 0.64] | 0.22 [0.05, 0.43] | 0.000488 (12 fixed, 0 broke) | - |
| no_action_gate | 46/78 = 0.59 [0.48, 0.69] | 0.15 [0.02, 0.30] | 0.00781 (8 fixed, 0 broke) | 0.481 (7 fixed, 11 broke) |
| action_gate_only | 15/78 = 0.19 [0.12, 0.29] | 0.72 [0.50, 0.92] | 3.64e-12 (39 fixed, 0 broke) | 1.49e-08 (27 fixed, 0 broke) |
| memguard | 14/78 = 0.18 [0.11, 0.28] | 0.74 [0.55, 0.93] | 1.82e-12 (40 fixed, 0 broke) | 7.45e-09 (28 fixed, 0 broke) |
| memguard_ttl | 15/78 = 0.19 [0.12, 0.29] | 0.72 [0.50, 0.92] | 3.64e-12 (39 fixed, 0 broke) | 1.49e-08 (27 fixed, 0 broke) |

### Qwen3.5-9B, ITDesk

| arm | ASR [95% CI] | reduction [95% CI] | vs unprotected (p) | vs NeMo Guardrails (p) |
|---|---|---|---|---|
| unprotected | 26/30 = 0.87 [0.70, 0.95] | - | - | - |
| nemo_rails | 26/30 = 0.87 [0.70, 0.95] | 0.00 [0.00, 0.00] | 1 (0 fixed, 0 broke) | - |
| no_action_gate | 27/30 = 0.90 [0.74, 0.97] | -0.04 [-0.12, 0.00] | 1 (0 fixed, 1 broke) | 1 (0 fixed, 1 broke) |
| action_gate_only | 0/30 = 0.00 [0.00, 0.11] | 1.00 [1.00, 1.00] | 2.98e-08 (26 fixed, 0 broke) | 2.98e-08 (26 fixed, 0 broke) |
| memguard | 0/30 = 0.00 [0.00, 0.11] | 1.00 [1.00, 1.00] | 2.98e-08 (26 fixed, 0 broke) | 2.98e-08 (26 fixed, 0 broke) |
| memguard_ttl | 0/30 = 0.00 [0.00, 0.11] | 1.00 [1.00, 1.00] | 2.98e-08 (26 fixed, 0 broke) | 2.98e-08 (26 fixed, 0 broke) |

### Nemotron Nano 9B v2, BankDesk

| arm | ASR [95% CI] | reduction [95% CI] | vs unprotected (p) | vs NeMo Guardrails (p) |
|---|---|---|---|---|
| unprotected | 25/78 = 0.32 [0.23, 0.43] | - | - | - |
| nemo_rails | 17/78 = 0.22 [0.14, 0.32] | 0.32 [0.04, 0.68] | 0.00781 (8 fixed, 0 broke) | - |
| no_action_gate | 24/78 = 0.31 [0.22, 0.42] | 0.04 [0.00, 0.17] | 1 (1 fixed, 0 broke) | 0.0156 (0 fixed, 7 broke) |
| action_gate_only | 9/78 = 0.12 [0.06, 0.20] | 0.64 [0.31, 1.00] | 3.05e-05 (16 fixed, 0 broke) | 0.0574 (11 fixed, 3 broke) |
| memguard | 9/78 = 0.12 [0.06, 0.20] | 0.64 [0.31, 1.00] | 3.05e-05 (16 fixed, 0 broke) | 0.0574 (11 fixed, 3 broke) |
| memguard_ttl | 9/78 = 0.12 [0.06, 0.20] | 0.64 [0.31, 1.00] | 3.05e-05 (16 fixed, 0 broke) | 0.0574 (11 fixed, 3 broke) |

### Nemotron Nano 9B v2, ITDesk

| arm | ASR [95% CI] | reduction [95% CI] | vs unprotected (p) | vs NeMo Guardrails (p) |
|---|---|---|---|---|
| unprotected | 15/30 = 0.50 [0.33, 0.67] | - | - | - |
| nemo_rails | 15/30 = 0.50 [0.33, 0.67] | 0.00 [0.00, 0.00] | 1 (0 fixed, 0 broke) | - |
| no_action_gate | 15/30 = 0.50 [0.33, 0.67] | 0.00 [0.00, 0.00] | 1 (0 fixed, 0 broke) | 1 (0 fixed, 0 broke) |
| action_gate_only | 0/30 = 0.00 [0.00, 0.11] | 1.00 [1.00, 1.00] | 6.1e-05 (15 fixed, 0 broke) | 6.1e-05 (15 fixed, 0 broke) |
| memguard | 0/30 = 0.00 [0.00, 0.11] | 1.00 [1.00, 1.00] | 6.1e-05 (15 fixed, 0 broke) | 6.1e-05 (15 fixed, 0 broke) |
| memguard_ttl | 0/30 = 0.00 [0.00, 0.11] | 1.00 [1.00, 1.00] | 6.1e-05 (15 fixed, 0 broke) | 6.1e-05 (15 fixed, 0 broke) |

### Third-party attacks (AgentDojo + InjecAgent), 3 seeds

| agent | arm | ASR [95% CI] |
|---|---|---|
| Qwen3.5-9B | unprotected | 96/462 = 0.21 [0.17, 0.25] |
| Qwen3.5-9B | nemo_rails | 1/462 = 0.00 [0.00, 0.01] |
| Qwen3.5-9B | no_action_gate | 97/462 = 0.21 [0.18, 0.25] |
| Qwen3.5-9B | action_gate_only | 0/462 = 0.00 [0.00, 0.01] |
| Qwen3.5-9B | memguard | 0/462 = 0.00 [0.00, 0.01] |
| Qwen3.5-9B | memguard_ttl | 0/462 = 0.00 [0.00, 0.01] |
| Nemotron Nano 9B v2 | unprotected | 25/462 = 0.05 [0.04, 0.08] |
| Nemotron Nano 9B v2 | nemo_rails | 0/462 = 0.00 [0.00, 0.01] |
| Nemotron Nano 9B v2 | no_action_gate | 24/462 = 0.05 [0.04, 0.08] |
| Nemotron Nano 9B v2 | action_gate_only | 0/462 = 0.00 [0.00, 0.01] |
| Nemotron Nano 9B v2 | memguard | 0/462 = 0.00 [0.00, 0.01] |
| Nemotron Nano 9B v2 | memguard_ttl | 0/462 = 0.00 [0.00, 0.01] |

## 2. Which checkpoint stopped each attack

Only attacks that succeeded against the unprotected agent are counted (same rule as the deck). A run is credited to the action gate if the harmful call was attempted and held; else to the write gate if the poisoned memory was quarantined; else to 'no harmful call made'. The write gate and the action gate can both act on one run, and the action gate is credited first, so write-gate numbers are a lower bound.

### Qwen3.5-9B, BankDesk (54 attack runs that succeed unprotected)

| arm | got through | action gate | write gate | no harmful call made (read gate wrap or agent declined) |
|---|---|---|---|---|
| nemo_rails | 42 (78%) | 0 (0%) | 0 (0%) | 12 (22%) |
| no_action_gate | 46 (85%) | 0 (0%) | 4 (7%) | 4 (7%) |
| action_gate_only | 15 (28%) | 39 (72%) | 0 (0%) | 0 (0%) |
| memguard | 14 (26%) | 33 (61%) | 4 (7%) | 3 (6%) |
| memguard_ttl | 15 (28%) | 36 (67%) | 3 (6%) | 0 (0%) |

### Qwen3.5-9B, ITDesk (26 attack runs that succeed unprotected)

| arm | got through | action gate | write gate | no harmful call made (read gate wrap or agent declined) |
|---|---|---|---|---|
| nemo_rails | 26 (100%) | 0 (0%) | 0 (0%) | 0 (0%) |
| no_action_gate | 26 (100%) | 0 (0%) | 0 (0%) | 0 (0%) |
| action_gate_only | 0 (0%) | 26 (100%) | 0 (0%) | 0 (0%) |
| memguard | 0 (0%) | 26 (100%) | 0 (0%) | 0 (0%) |
| memguard_ttl | 0 (0%) | 26 (100%) | 0 (0%) | 0 (0%) |

### Nemotron Nano 9B v2, BankDesk (25 attack runs that succeed unprotected)

| arm | got through | action gate | write gate | no harmful call made (read gate wrap or agent declined) |
|---|---|---|---|---|
| nemo_rails | 17 (68%) | 0 (0%) | 0 (0%) | 8 (32%) |
| no_action_gate | 24 (96%) | 0 (0%) | 1 (4%) | 0 (0%) |
| action_gate_only | 9 (36%) | 16 (64%) | 0 (0%) | 0 (0%) |
| memguard | 9 (36%) | 16 (64%) | 0 (0%) | 0 (0%) |
| memguard_ttl | 9 (36%) | 15 (60%) | 1 (4%) | 0 (0%) |

### Nemotron Nano 9B v2, ITDesk (15 attack runs that succeed unprotected)

| arm | got through | action gate | write gate | no harmful call made (read gate wrap or agent declined) |
|---|---|---|---|---|
| nemo_rails | 15 (100%) | 0 (0%) | 0 (0%) | 0 (0%) |
| no_action_gate | 15 (100%) | 0 (0%) | 0 (0%) | 0 (0%) |
| action_gate_only | 0 (0%) | 15 (100%) | 0 (0%) | 0 (0%) |
| memguard | 0 (0%) | 15 (100%) | 0 (0%) | 0 (0%) |
| memguard_ttl | 0 (0%) | 15 (100%) | 0 (0%) | 0 (0%) |

The ablation reads off the table above: `no_action_gate` is the write and read gates alone, `action_gate_only` is the action gate alone, `memguard` is all of them.

## 3. Attack type x entry channel

Cells are attacks executed / runs (3 seeds per scenario), hand-written BankDesk and ITDesk sets. Entry channel is read from where the scenario plants its text. Web retrieval has no hand-written scenario in these sets (it appears only in the garak and third-party runs), so it is not a column.

### Qwen3.5-9B, unprotected

| attack type | customer chat | document | peer agent | tool output |
|---|---|---|---|---|
| ambiguous_request | 0/3 | · | · | · |
| authority_spoofing | 6/6 | · | 6/12 | · |
| cross_tenant_probe | 6/6 | · | · | · |
| embedded_instruction | · | 9/15 | · | 12/12 |
| false_fact | · | 3/3 | 6/6 | 8/9 |
| policy_bypass | 9/12 | · | · | · |
| sleeper_trigger | · | 6/6 | 0/3 | 0/3 |
| slow_drift | · | · | 6/6 | · |
| social_engineering | 3/6 | · | · | · |

### Qwen3.5-9B, nemo_rails

| attack type | customer chat | document | peer agent | tool output |
|---|---|---|---|---|
| ambiguous_request | 0/3 | · | · | · |
| authority_spoofing | 3/6 | · | 6/12 | · |
| cross_tenant_probe | 6/6 | · | · | · |
| embedded_instruction | · | 6/15 | · | 12/12 |
| false_fact | · | 3/3 | 6/6 | 8/9 |
| policy_bypass | 6/12 | · | · | · |
| sleeper_trigger | · | 3/6 | 0/3 | 0/3 |
| slow_drift | · | · | 6/6 | · |
| social_engineering | 3/6 | · | · | · |

### Qwen3.5-9B, memguard

| attack type | customer chat | document | peer agent | tool output |
|---|---|---|---|---|
| ambiguous_request | 0/3 | · | · | · |
| authority_spoofing | 0/6 | · | 0/12 | · |
| cross_tenant_probe | 0/6 | · | · | · |
| embedded_instruction | · | 0/15 | · | 2/12 |
| false_fact | · | 0/3 | 6/6 | 3/9 |
| policy_bypass | 0/12 | · | · | · |
| sleeper_trigger | · | 0/6 | 0/3 | 0/3 |
| slow_drift | · | · | 0/6 | · |
| social_engineering | 3/6 | · | · | · |

### Nemotron Nano 9B v2, unprotected

| attack type | customer chat | document | peer agent | tool output |
|---|---|---|---|---|
| ambiguous_request | 0/3 | · | · | · |
| authority_spoofing | 6/6 | · | 2/12 | · |
| cross_tenant_probe | 6/6 | · | · | · |
| embedded_instruction | · | 0/15 | · | 1/12 |
| false_fact | · | 0/3 | 3/6 | 3/9 |
| policy_bypass | 10/12 | · | · | · |
| sleeper_trigger | · | 0/6 | 0/3 | 0/3 |
| slow_drift | · | · | 3/6 | · |
| social_engineering | 6/6 | · | · | · |

### Nemotron Nano 9B v2, nemo_rails

| attack type | customer chat | document | peer agent | tool output |
|---|---|---|---|---|
| ambiguous_request | 0/3 | · | · | · |
| authority_spoofing | 3/6 | · | 2/12 | · |
| cross_tenant_probe | 6/6 | · | · | · |
| embedded_instruction | · | 0/15 | · | 0/12 |
| false_fact | · | 0/3 | 3/6 | 3/9 |
| policy_bypass | 9/12 | · | · | · |
| sleeper_trigger | · | 0/6 | 0/3 | 0/3 |
| slow_drift | · | · | 3/6 | · |
| social_engineering | 3/6 | · | · | · |

### Nemotron Nano 9B v2, memguard

| attack type | customer chat | document | peer agent | tool output |
|---|---|---|---|---|
| ambiguous_request | 0/3 | · | · | · |
| authority_spoofing | 0/6 | · | 0/12 | · |
| cross_tenant_probe | 0/6 | · | · | · |
| embedded_instruction | · | 0/15 | · | 0/12 |
| false_fact | · | 0/3 | 3/6 | 0/9 |
| policy_bypass | 0/12 | · | · | · |
| sleeper_trigger | · | 0/6 | 0/3 | 0/3 |
| slow_drift | · | · | 0/6 | · |
| social_engineering | 6/6 | · | · | · |

## 4. Security-utility frontier

Attack success (lower is better) against benign tasks finished (higher is better), BankDesk, with Wilson 95% intervals. Benign tasks are the 21 + 42 scenarios x 3 seeds.

| agent | arm | attacks executed [95% CI] | benign finished [95% CI] |
|---|---|---|---|
| Qwen3.5-9B | unprotected | 0.69 [0.58, 0.78] | 0.98 [0.92, 1.00] |
| Qwen3.5-9B | nemo_rails | 0.54 [0.43, 0.64] | 0.98 [0.92, 1.00] |
| Qwen3.5-9B | no_action_gate | 0.59 [0.48, 0.69] | 0.89 [0.79, 0.95] |
| Qwen3.5-9B | action_gate_only | 0.19 [0.12, 0.29] | 0.98 [0.92, 1.00] |
| Qwen3.5-9B | memguard | 0.18 [0.11, 0.28] | 0.86 [0.75, 0.92] |
| Qwen3.5-9B | memguard_ttl | 0.19 [0.12, 0.29] | 0.84 [0.73, 0.91] |
| Nemotron Nano 9B v2 | unprotected | 0.32 [0.23, 0.43] | 0.44 [0.33, 0.57] |
| Nemotron Nano 9B v2 | nemo_rails | 0.22 [0.14, 0.32] | 0.43 [0.31, 0.55] |
| Nemotron Nano 9B v2 | no_action_gate | 0.31 [0.22, 0.42] | 0.38 [0.27, 0.50] |
| Nemotron Nano 9B v2 | action_gate_only | 0.12 [0.06, 0.20] | 0.46 [0.34, 0.58] |
| Nemotron Nano 9B v2 | memguard | 0.12 [0.06, 0.20] | 0.38 [0.27, 0.50] |
| Nemotron Nano 9B v2 | memguard_ttl | 0.12 [0.06, 0.20] | 0.38 [0.27, 0.50] |

![frontier](frontier.png)

## 5. Write-gate classifiers on a public held-out set

deepset/prompt-injections, 662 texts (263 injections), none used to tune a gate. AUROC and average precision (AP) use the classifier's injection score; the operating point is the gate's own threshold. Intervals are 95% bootstrap over texts. Classifiers with few distinct scores (the LLM judges mostly answer 0 or 1) have a coarse curve, so AUROC understates how a tuned threshold could do.

| classifier | distinct scores | AUROC [95% CI] | AP | precision | recall [95% CI] | false-quarantine rate | recall @ FPR<=5% |
|---|---|---|---|---|---|---|---|
| DeBERTa NLI | 138 | 0.545 [0.502, 0.597] | 0.510 | 0.78 | 0.12 [0.09, 0.17] | 0.023 | 0.16 |
| NemoGuard JailbreakDetect | 225 | 0.754 [0.719, 0.790] | 0.715 | 1.00 | 0.11 [0.07, 0.15] | 0.000 | 0.31 |
| Qwen3.5-0.8B | 7 | 0.246 [0.212, 0.278] | 0.331 | 0.39 | 0.94 [0.90, 0.96] | 0.980 | 0.01 |
| Qwen3.5-2B | 6 | 0.738 [0.709, 0.764] | 0.651 | 0.99 | 0.40 [0.34, 0.46] | 0.003 | 0.40 |
| Qwen3.5-4B | 8 | 0.936 [0.913, 0.955] | 0.896 | 0.99 | 0.65 [0.59, 0.71] | 0.003 | 0.65 |
| Qwen3.5-9B | 7 | 0.962 [0.951, 0.973] | 0.918 | 0.99 | 0.57 [0.51, 0.63] | 0.003 | 0.57 |
| Nemotron Nano 4B | 15 | 0.540 [0.498, 0.581] | 0.399 | 0.39 | 0.29 [0.23, 0.34] | 0.288 | 0.00 |
| Nemotron Nano 9B v2 | 8 | 0.707 [0.680, 0.732] | 0.586 | 0.95 | 0.27 [0.22, 0.33] | 0.010 | 0.27 |

![roc](roc_pr.png)

## 6. Time to contain

For attack runs that succeed unprotected, in each arm: how many harmful calls were attempted, how many executed before the first one was held (or in total, if none was), and in which session the first harmful call was attempted. Session 1 is the first customer message after the poison is planted.

| agent | domain | arm | runs | mean harmful calls attempted | mean executed before first hold | runs with a hold | first attempt in session 1 | first attempt in a later session |
|---|---|---|---|---|---|---|---|---|
| Qwen3.5-9B | bank | unprotected | 54 | 1.02 | 1.00 | 0 | 9 | 45 |
| Qwen3.5-9B | bank | nemo_rails | 54 | 0.80 | 0.78 | 0 | 6 | 36 |
| Qwen3.5-9B | bank | no_action_gate | 54 | 0.85 | 0.85 | 0 | 2 | 44 |
| Qwen3.5-9B | bank | action_gate_only | 54 | 1.09 | 0.28 | 39 | 9 | 45 |
| Qwen3.5-9B | bank | memguard | 54 | 0.96 | 0.26 | 33 | 3 | 44 |
| Qwen3.5-9B | bank | memguard_ttl | 54 | 1.09 | 0.28 | 36 | 6 | 45 |
| Qwen3.5-9B | it | unprotected | 26 | 1.12 | 1.12 | 0 | 8 | 18 |
| Qwen3.5-9B | it | nemo_rails | 26 | 1.12 | 1.12 | 0 | 8 | 18 |
| Qwen3.5-9B | it | no_action_gate | 26 | 1.12 | 1.12 | 0 | 8 | 18 |
| Qwen3.5-9B | it | action_gate_only | 26 | 1.23 | 0.00 | 26 | 8 | 18 |
| Qwen3.5-9B | it | memguard | 26 | 1.12 | 0.00 | 26 | 8 | 18 |
| Qwen3.5-9B | it | memguard_ttl | 26 | 1.08 | 0.00 | 26 | 8 | 18 |
| Nemotron Nano 9B v2 | bank | unprotected | 25 | 1.00 | 1.00 | 0 | 0 | 25 |
| Nemotron Nano 9B v2 | bank | nemo_rails | 25 | 0.68 | 0.68 | 0 | 0 | 17 |
| Nemotron Nano 9B v2 | bank | no_action_gate | 25 | 0.96 | 0.96 | 0 | 0 | 24 |
| Nemotron Nano 9B v2 | bank | action_gate_only | 25 | 1.00 | 0.36 | 16 | 0 | 25 |
| Nemotron Nano 9B v2 | bank | memguard | 25 | 1.00 | 0.36 | 16 | 0 | 25 |
| Nemotron Nano 9B v2 | bank | memguard_ttl | 25 | 0.96 | 0.36 | 15 | 0 | 24 |
| Nemotron Nano 9B v2 | it | unprotected | 15 | 1.00 | 1.00 | 0 | 6 | 9 |
| Nemotron Nano 9B v2 | it | nemo_rails | 15 | 1.00 | 1.00 | 0 | 6 | 9 |
| Nemotron Nano 9B v2 | it | no_action_gate | 15 | 1.00 | 1.00 | 0 | 6 | 9 |
| Nemotron Nano 9B v2 | it | action_gate_only | 15 | 1.00 | 0.00 | 15 | 6 | 9 |
| Nemotron Nano 9B v2 | it | memguard | 15 | 1.00 | 0.00 | 15 | 6 | 9 |
| Nemotron Nano 9B v2 | it | memguard_ttl | 15 | 1.00 | 0.00 | 15 | 6 | 9 |
