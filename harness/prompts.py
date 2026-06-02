# Meta-Harness prompt 构造模块。
# 负责为 solver、verifier 和 DeepAgent proposer 生成结构化提示词。

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from harness.memory import Example, Prediction, RetrievedExample
from harness.memory import format_choices


# 构造文本分类 solver 的 prompt。
# 会把当前样本、候选标签和检索到的训练样本组织成 chat messages。
def build_classifier_prompt(
    example: Example,
    retrieved: Sequence[RetrievedExample],
) -> list[dict[str, str]]:
    # 将检索样本格式化为 few-shot 示例块。
    fewshot = "\n\n".join(
        f"Example {idx + 1}\nText: {item.text}\nLabel: {item.label}"
        for idx, item in enumerate(retrieved)
    )
    user = "Follow the task instructions in the prompt and return the answer as JSON.\n"
    if example.choices:
        user += f"Choices: {format_choices(example.choices)}\n\n"
    else:
        user += "\n"
    if fewshot:
        user += f"Relevant training examples:\n{fewshot}\n\n"
    user += f"Task prompt:\n{example.text}\nReturn JSON: {{\"final_answer\": \"...\"}}"
    return [
        {"role": "system", "content": "You are a careful text classifier."},
        {"role": "user", "content": user},
    ]


# 构造二次验证 prompt。
# verifier 会参考原任务和第一次预测，重新给出 final_answer。
def build_verification_prompt(
    example: Example,
    first_prediction: Prediction,
) -> list[dict[str, str]]:
    # 保留候选标签，避免 verifier 输出不在标签集合内的答案。
    user = "Verify whether the first prediction is correct for the task.\n"
    if example.choices:
        user += f"Choices: {format_choices(example.choices)}\n"
    user += (
        f"Task prompt:\n{example.text}\n"
        f"First prediction: {first_prediction.label}\n"
        "Return JSON: {\"final_answer\": \"...\"}"
    )
    return [
        {"role": "system", "content": "You verify text classification decisions."},
        {"role": "user", "content": user},
    ]


# 构造 DeepAgent proposer 的完整任务 prompt。
# 该 prompt 告诉 proposer 当前实验状态、候选输出协议、可检查文件和必须遵守的工作流。
def build_proposer_prompt(
    frontier: Mapping[str, Any],
    leaderboard: Sequence[Mapping[str, Any]],
    recent_traces: Sequence[Mapping[str, Any]],
    candidate_contract: Mapping[str, Any],
    source_files: Mapping[str, str] | None = None,
    dataset_specs: Sequence[Mapping[str, Any]] | None = None,
    result_summaries: Sequence[Mapping[str, Any]] | None = None,
    artifact_paths: Sequence[str] | None = None,
) -> str:
    # 先把路径、trace 和结果摘要压缩成可读索引，避免 prompt 过长。
    source_index = _format_source_files(source_files or {})
    state_paths = _format_path_list(artifact_paths or [])
    trace_index = _format_json_index(recent_traces)
    result_index = _format_json_index(result_summaries or [])

    # 主体 prompt 强制 proposer 先分析、再原型验证、再写候选，最后登记 pending_eval.json。
    return (
        "You are DeepAgent acting as the Meta-Harness proposer for text classification.\n"
        "Do all work in the main session. Do not delegate.\n"
        "Do not run the full benchmark. The outer loop evaluates candidates.\n"
        "You do NOT run benchmarks.\n"
        "You MUST follow the workflow below in order. Do not skip a step.\n\n"
        "WORKFLOW\n"
        "Step 0: Post-eval reports\n"
        "- Check the reports directory in the run directory.\n"
        "- For each past iteration that has results in evolution_summary.jsonl but no report, write one.\n\n"
        "Step 1: Analyze\n"
        "- You MUST read all state files if they exist before proposing.\n"
        "- MUST read if present: evolution_summary.jsonl, frontier_val.json, config.yaml, recent logs/<dataset>/<agent>/<model>/log.jsonl.\n"
        "- In this project, the same logs are stored under logs/<dataset>/<memory>/<model>/log.jsonl.\n"
        "- SHOULD inspect: leaderboard.csv, relevant val.json files, current baseline/top/Pareto source files.\n"
        "- Do not rely on the prompt summary alone; inspect the files yourself.\n\n"
        "Step 2: Prototype — MANDATORY\n"
        "- Before writing final candidates, prototype the core retrieval/learning idea in /tmp/.\n"
        "- Use real examples from the logs/ traces to test variants.\n"
        "- Try 2-3 variants and compare before choosing the final mechanism.\n"
        "- Delete temporary scripts when done.\n\n"
        "Step 3: Implement\n"
        "- Write exactly 2 candidates.\n"
        "- Copy a strong existing base system, then make targeted modifications.\n"
        "- Improve one or more of these axes: what training examples or experience to store, how to retrieve relevant memories, how to organize the classification prompt, whether and how to perform second-pass verification.\n"
        "- Do not edit config.yaml just to register candidates. The benchmark auto-discovers files in agents/.\n"
        "- After implementation, self-critique: if the logic is only parameter changes, rewrite it with a real mechanism change.\n\n"
        "Step 4: Write pending_eval.json\n"
        "- Only after Steps 0-3 are complete, write pending_eval.json.\n"
        "- Do not write pending_eval.json early.\n"
        "- In the final response, briefly list the state/result/trace files you inspected.\n\n"
        "The evaluation contract is fixed: dataset loading, prompt wrapping, target extraction, and accuracy calculation are handled by the harness and are not optimization targets.\n"
        "Improve only the memory-system candidate logic.\n\n"
        "Write exactly two candidate Python files under agents/.\n"
        "Each candidate must define a BaseAgentMemory subclass implementing predict(input), learn_from_batch(batch_results), get_state(), and set_state().\n"
        "Write pending_eval.json to the path specified in Candidate contract.\n"
        "Do not edit or depend on evaluator, data loader, run-loop, or benchmark implementation code.\n"
        "Source contents are not inlined; inspect the full files by path when needed.\n\n"
        "Return a short final message naming the candidates and the files you inspected.\n\n"
        f"Candidate contract:\n{json.dumps(dict(candidate_contract), indent=2, ensure_ascii=False)}\n\n"
        f"State file paths to inspect:\n{state_paths}\n\n"
        f"Result file index:\n{result_index}\n\n"
        f"Trace file index:\n{trace_index}\n\n"
        f"Source file index:\n{source_index}\n"
    )


# 格式化源码文件索引。
# 这里只列路径，不内联源码内容，让 DeepAgent 按需读取完整文件。
def _format_source_files(source_files: Mapping[str, str]) -> str:
    if not source_files:
        return "(no source files provided)\n"
    parts = []
    for path in source_files:
        parts.append(f"- {path}")
    return "\n".join(parts) + "\n"


# 格式化当前 run 目录下的状态文件路径。
def _format_path_list(paths: Sequence[str]) -> str:
    if not paths:
        return "(no state files found yet)\n"
    return "\n".join(f"- {path}" for path in paths) + "\n"


# 压缩 trace/result 索引，只保留 proposer 定位文件和分析结果所需字段。
def _format_json_index(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return "(none yet)\n"
    compact_rows = []
    for row in rows:
        # 过滤掉大字段，避免把完整 trace 或无关内容塞进 proposer prompt。
        compact_rows.append(
            {
                key: value
                for key, value in row.items()
                if key
                in {
                    "name",
                    "system",
                    "dataset",
                    "path",
                    "split",
                    "log_hint",
                    "val_hint",
                    "read_trace_file",
                    "total",
                    "wrong",
                }
            }
        )
    return json.dumps(compact_rows, indent=2, ensure_ascii=False) + "\n"
