# Implementation Spec

This is the current build contract for `tc_deepagent_metaharness`.

## Protocol

The main memory-system protocol is the local official-like agent protocol in `harness/agent_protocol.py`.

Required candidate interface:

```python
class MyMemory(BaseAgentMemory):
    def predict(self, input: str) -> tuple[str, dict[str, Any]]: ...
    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None: ...
    def get_state(self) -> str: ...
    def set_state(self, state: str) -> None: ...
```

Candidates are written under `agents/`. The main baselines are:

- `agents/no_memory.py`
- `agents/fewshot_all.py`

The older compact `systems/` protocol remains only for historical tests and compatibility helpers; it is not the main harness protocol.

## Evaluation Semantics

`harness/official_eval.py` is self-written but follows the local official text-classification behavior:

- load train/val/test rows with the same prompt wrapping and shuffle order;
- train offline by batching rows into `learn_from_batch()`;
- save val-trained memory state to `logs/{dataset}/{memory}/{model}/memory.json`;
- during test, load that saved memory state instead of retraining;
- write val artifacts under `logs/...`;
- write test artifacts under `results/...`;
- use evaluator extraction based on `final_answer`, not the wider compact `extract_label()` fallback.

## Proposer Boundary

DeepAgent replaces Claude Code only as proposer. It may write:

- `agents/{candidate}.py`
- `runs/{run}/pending_eval.json`
- `runs/{run}/iter_XXX/proposer_tool_calls.jsonl`
- `runs/{run}/iter_XXX/proposer_messages.json`
- `runs/{run}/iter_XXX/proposer_response.md`
- optional `runs/{run}/reports/*.md`

It should inspect state by path:

- `evolution_summary.jsonl`
- `frontier_val.json`
- `config.yaml`
- `logs/{dataset}/{memory}/{model}/log.jsonl`
- `logs/{dataset}/{memory}/{model}/val.json`

The prompt/source index must not inline evaluator, data loader, benchmark, run-loop, or proposer implementation source.

Every real proposer call must record the proposer's actual backend activity. `proposer_tool_calls.jsonl`
contains one JSON line per backend action, including `read`, `write`, `edit`, `ls`, `glob`, `grep`,
and `execute` calls, with arguments and bounded result summaries.

## Frontier And Final Test

The frontier is Pareto over:

- maximize validation accuracy;
- minimize context length.

Final test evaluates:

- all baselines;
- every Pareto system;
- each dataset's best validation system.

## Key Files

- `harness/agent_protocol.py`: local official-like memory base class, JSON extraction, dynamic agent loader.
- `agents/no_memory.py`: direct prompt baseline.
- `agents/fewshot_all.py`: few-shot-all baseline with official-like sampling/shuffle/context budget.
- `harness/official_eval.py`: self-written official-like evaluator and artifact writer.
- `harness/loop.py`: baseline eval, proposer loop, Pareto frontier, final test scheduling.
- `harness/proposer.py`: DeepAgent integration and dry-run candidate generation.
- `harness/prompts.py`: solver and proposer prompt builders.
- `harness/store.py`: run logs and artifact helpers.
- `harness/cli.py`: `benchmark`, `run`, `preflight`, `report`.

## Test Discipline

Run focused tests after each unit, then full tests:

```bash
uv run pytest tests/test_agent_protocol.py tests/test_official_eval.py -q
uv run pytest tests/test_proposer.py tests/test_loop.py tests/test_cli.py -q
uv run pytest -q
```
