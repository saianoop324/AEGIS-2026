# Third-party test cases: AgentDojo and InjecAgent (planned, not built)

Every attack set so far was written or framed by us, after reading the gates' code (Experiments 4–14).
The goal here is a test set written by people who never saw MemGuard: two published prompt-injection
benchmarks for tool-using agents, converted into BankDesk scenarios and run through
`experiments/run_bank_scenarios.py` like the other sets.

Status (26 Oct 2026): the benchmarks were installed and surveyed on Mriya; the converter is not written
yet, so nothing has been run.

## What is set up on Mriya

- Conda env `thirdparty` with `agentdojo` 0.1.35.
- `~/thirdparty/InjecAgent`: clone of https://github.com/uiuc-kang-lab/InjecAgent (data in `data/`).

## AgentDojo, banking suite v1.2.1

Debenedetti et al., NeurIPS 2024 Datasets & Benchmarks. https://github.com/ethz-spylab/agentdojo

- 16 user tasks, 9 injection tasks. 8 of the 9 injection goals end in money reaching the attacker's
  account (AgentDojo tools `send_money`, `update_scheduled_transaction`); one changes the user's
  password, which BankDesk has no tool for, so it would be skipped.
- 4 injection points: an incoming transaction's subject, and three files (a bill, a landlord notice,
  an address-change letter).
- Injected text is produced by AgentDojo's own attack classes (`agentdojo.attacks.attack_registry`;
  e.g. `important_instructions`, `ignore_previous`, `injecagent`). `get_injection_candidates(user_task)`
  tells which injection points a user task exposes.

Mapping onto BankDesk:

| AgentDojo | BankDesk |
|---|---|
| file in the filesystem | `setup.documents` entry (doc_id = file name) |
| incoming transaction subject | `setup.transactions` entry, subject as `memo` |
| user task prompt | the scenario's session message |
| `send_money` / scheduled transaction to the attacker's IBAN | oracle: `transfer` or `add_payee` whose account number contains the IBAN's digits |
| user name in attack templates | the verified BankDesk customer, Asha Rao |

## InjecAgent, direct-harm base set

Zhan et al., Findings of ACL 2024. `data/test_cases_dh_base.json` (510 cases).

- 34 cases have a bank operation as the attacker tool (`BankManagerTransferFunds`,
  `BankManagerPayBill`): 2 attacker goals, each placed in the tool output of 17 different user tools.
- Each case's `Tool Response` already contains the injection. It would become an uploaded document
  the user asks the agent to look at; the oracle is a `transfer` or `add_payee` to the account (or
  payee) named in the attacker goal.

## How to run, once the converter exists

Write the converted scenarios to `experiments/bank_scenarios_thirdparty.json` in the existing format
(see the docstring of `experiments/run_bank_scenarios.py`), then run with the Experiment 14 settings,
e.g. on Mriya:

```bash
python experiments/run_bank_scenarios.py experiments/bank_scenarios_thirdparty.json --seeds 3 --persistence-sessions 3 --arms unprotected,nemo_rails,action_gate_only,no_action_gate,memguard,memguard_ttl --out experiments/exp15_deberta_bank_scenarios_thirdparty.json
```

The results then show up in the dashboard's experiments tab and can be rescored with
`experiments/rescore_results.py`.
