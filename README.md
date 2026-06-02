# tc_deepagent_metaharness

Compact self-written Meta-Harness reproduction for text classification.

The text-classification solver is `gpt-oss-120b`. DeepAgent is only the proposer; its model is `claude-opus-4-6` through the Doro Claude Code proxy. Candidate memory systems are written under `agents/`.

## Setup

```bash
cd /home/wyqdf/harness/tc_deepagent_metaharness
uv sync
```

Plaintext API values are supported in config files. Environment overrides are also supported:

- solver: `SOLVER_API_KEY`, `SOLVER_BASE_URL=https://9527code.com/v1`
- proposer: `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_BASE_URL=https://doro.lol`

## Commands

```bash
uv run pytest -q
uv run python -m harness.cli benchmark --system agents/fewshot_all.py --split val --stub
uv run python -m harness.cli run --config configs/dry_run.yaml
uv run python -m harness.cli run --config configs/oss120b_deepagent_opus46_10rounds_official.yaml
uv run python -m harness.cli report --run runs/oss120b_deepagent_opus46_10rounds_official_agents_trace_20260524
```

## Candidate Contract

Each candidate file must define a subclass of `harness.agent_protocol.BaseAgentMemory`:

```python
class Candidate(BaseAgentMemory):
    def predict(self, input: str) -> tuple[str, dict]:
        ...

    def learn_from_batch(self, batch_results: list[dict]) -> None:
        ...

    def get_state(self) -> str:
        ...

    def set_state(self, state: str) -> None:
        ...
```

The main baselines are `agents/no_memory.py` and `agents/fewshot_all.py`. Historical `systems/` files are kept for old compact tests, but new harness runs, proposer output, frontier, and final test use `agents/`.

## Run Artifacts

Official-like artifacts are written under each run:

```text
runs/{run_name}/
  config.yaml
  live.log
  latest.log
  leaderboard.csv
  frontier.json
  frontier_val.json
  baseline_scores.json
  final_test_scores.json
  pending_eval.json
  logs/{dataset}/{memory}/{model}/
    val.json
    log.jsonl
    memory.json
  results/{dataset}/{memory}/{model}/
    test.json
    log.jsonl
  iter_001/
    proposer_prompt.md
    proposer_tool_calls.jsonl
    proposer_messages.json
    proposer_response.md
    pending_eval.json
    {candidate}.py
    {candidate}_scores.json
```

`proposer_tool_calls.jsonl` is the per-round DeepAgent transcript. It records backend-level
file reads/writes/searches and shell commands with arguments and bounded result previews.
