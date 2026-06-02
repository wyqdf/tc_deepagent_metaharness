# Project Structure

This project is a compact, self-written reproduction of the Meta-Harness text-classification loop. It keeps the official-like evaluation protocol and replaces only the proposer with DeepAgent.

```text
tc_deepagent_metaharness/
  README.md
  IMPLEMENTATION_SPEC.md
  PROJECT_STRUCTURE.md
  pyproject.toml
  .env.example

  configs/
    dry_run.yaml
    experiment.yaml
    oss120b_deepagent_opus46_10rounds_official.yaml

  agents/
    __init__.py
    no_memory.py
    fewshot_all.py
    {deepagent_candidate}.py

  harness/
    __init__.py
    agent_protocol.py
    cli.py
    data.py
    eval.py
    evaluators.py
    llm.py
    loop.py
    memory.py
    metrics.py
    official_eval.py
    prompts.py
    proposer.py
    store.py

  systems/
    baseline.py
    fewshot_all.py
    generated/

  runs/
    {run_name}/

  tests/
    test_agent_protocol.py
    test_official_eval.py
    test_loop.py
    ...
```

## Main Path

- `agents/` is the main candidate directory.
- `harness/agent_protocol.py` defines the official-like memory interface.
- `harness/official_eval.py` evaluates `agents/*.py` with offline train, val memory persistence, and test memory reload.
- `harness/loop.py` runs baselines, calls DeepAgent, validates candidates, updates Pareto frontier, and runs final test.
- `harness/proposer.py` writes dry-run and DeepAgent candidates under `agents/`.

## Compatibility Path

`systems/` is retained for older compact-protocol tests and comparison helpers. New runs do not use it as the candidate directory, and `systems/generated/` is excluded from the proposer source index, frontier, and final test.

## Writable Areas For Proposer

DeepAgent should write only:

- `agents/{candidate}.py`
- `runs/{run_name}/pending_eval.json`
- optional `runs/{run_name}/reports/*.md`

The outer loop handles import validation, benchmark evaluation, frontier updates, and final test scheduling.

## Run Artifacts

```text
runs/{run_name}/
  live.log
  latest.log
  config.yaml
  leaderboard.csv
  frontier_val.json
  frontier.json
  evolution.jsonl
  evolution_summary.jsonl
  pending_eval.json
  logs/{dataset}/{memory}/{model}/val.json
  logs/{dataset}/{memory}/{model}/log.jsonl
  logs/{dataset}/{memory}/{model}/memory.json
  results/{dataset}/{memory}/{model}/test.json
  results/{dataset}/{memory}/{model}/log.jsonl
```
