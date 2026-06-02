"""Tests for official-semantics evaluation."""
# Regression checks for the harness contract continue below.


from __future__ import annotations

import json
import random
import sys
import types
from pathlib import Path

from harness.eval import DatasetSpec
from harness.llm import StubLLM
from harness.official_eval import (
    evaluate_memory_official,
    evaluate_system_official,
    load_dataset_splits_3way_from_specs,
)
from harness.agent_protocol import load_agent_memory


def _install_reference_import_stubs() -> None:
    if "datasets" not in sys.modules:
        datasets_stub = types.ModuleType("datasets")
        datasets_stub.load_dataset = lambda *args, **kwargs: None
        sys.modules["datasets"] = datasets_stub
    if "litellm" not in sys.modules:
        litellm_stub = types.ModuleType("litellm")
        litellm_stub.completion = lambda *args, **kwargs: None
        litellm_stub.completion_cost = lambda *args, **kwargs: 0.0
        litellm_stub.token_counter = lambda *args, **kwargs: 0
        sys.modules["litellm"] = litellm_stub
    if "openai_harmony" not in sys.modules:
        harmony_stub = types.ModuleType("openai_harmony")

        class _HarmonyEncodingName:
            HARMONY_GPT_OSS = "HARMONY_GPT_OSS"

        class _Role:
            ASSISTANT = "assistant"

        class _Encoding:
            def encode(self, raw_content, allowed_special="all"):
                return []

            def parse_messages_from_completion_tokens(self, tokens, role=None, strict=False):
                return []

        harmony_stub.HarmonyEncodingName = _HarmonyEncodingName
        harmony_stub.Role = _Role
        harmony_stub.load_harmony_encoding = lambda *args, **kwargs: _Encoding()
        sys.modules["openai_harmony"] = harmony_stub


def test_official_split_loader_matches_reference_shuffle_order():
    reference_root = Path("/home/wyqdf/harness/meta-harness/reference_examples")
    if str(reference_root) not in sys.path:
        sys.path.insert(0, str(reference_root))
    _install_reference_import_stubs()

    from text_classification.data import load_dataset_splits_3way

    dataset = DatasetSpec(
        name="USPTO",
        train_path="/home/wyqdf/harness/meta-harness/reference_examples/text_classification/data/uspto/train.jsonl",
        val_path="/home/wyqdf/harness/meta-harness/reference_examples/text_classification/data/uspto/val.jsonl",
        test_path="/home/wyqdf/harness/meta-harness/reference_examples/text_classification/data/uspto/test.jsonl",
    )

    ours = load_dataset_splits_3way_from_specs(
        dataset,
        seed=42,
        limits={"USPTO": {"num_train": 5, "num_val": 3, "num_test": 4}},
    )
    reference = load_dataset_splits_3way(
        "USPTO",
        num_train=5,
        num_val=3,
        num_test=4,
        shuffle_seed=42,
    )

    for our_split, ref_split in zip(ours, reference[:3]):
        assert our_split == ref_split


def test_official_eval_uses_reference_offline_train_and_split_semantics(tmp_path):
    project = tmp_path
    (project / "systems").mkdir()
    (project / "data/demo").mkdir(parents=True)
    system_path = project / "systems/counter_memory.py"
    system_path.write_text(
        """
from harness.memory import Example, Prediction

class CounterMemory:
    name = "counter_memory"
    def __init__(self):
        self.examples = []
    def learn(self, examples, llm):
        self.examples = list(examples)
    def predict(self, example, llm):
        return Prediction(self.examples[-1].label, self.examples[-1].label, [], [])

def build_memory_system(config=None):
    return CounterMemory()
""".strip()
        + "\n",
        encoding="utf-8",
    )
    for split, labels in {
        "train": ["A", "B", "C"],
        "val": ["B", "C"],
        "test": ["C", "A"],
    }.items():
        rows = [
            {"id": f"{split}-{idx}", "text": f"{split} text {idx}", "label": label, "choices": ["A", "B", "C"]}
            for idx, label in enumerate(labels)
        ]
        (project / f"data/demo/{split}.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )
    dataset = DatasetSpec(
        "demo",
        str(project / "data/demo/train.jsonl"),
        str(project / "data/demo/val.jsonl"),
        str(project / "data/demo/test.jsonl"),
    )
    shuffled_train = ["A", "B", "C"]
    random.Random(42).shuffle(shuffled_train)
    expected_prediction = shuffled_train[-1]

    result = evaluate_system_official(
        system_path,
        [dataset],
        "val",
        StubLLM(),
        {"seed": 42, "dataset_limits": {"demo": {"num_train": 3, "num_val": 2, "num_test": 2}}},
        max_workers=2,
    )

    assert [row["prediction"] for row in result.traces] == [expected_prediction, expected_prediction]
    assert result.summary["total"] == 2
    assert result.per_dataset == {"demo": 0.5}


def test_official_eval_counts_single_prediction_failure_as_incorrect():
    class FlakyMemory:
        def __init__(self):
            self.prompt = ""

        def predict(self, input):
            self.prompt = input
            if input == "bad":
                raise RuntimeError("temporary connection failure")
            return "A", {"full_response": "A"}

        def get_last_prompt_info(self):
            return {
                "prompt_len": len(self.prompt),
                "prompt_hash": "hash",
                "prompt_text": self.prompt,
            }

    rows = [
        {"input": "good", "target": "A"},
        {"input": "bad", "target": "A"},
    ]

    result = evaluate_memory_official(
        FlakyMemory(),
        rows,
        lambda prediction, target, **kwargs: prediction == target,
        "demo",
        max_workers=2,
    )

    assert len(result) == 2
    assert result[0]["was_correct"] is True
    assert result[1]["was_correct"] is False
    assert result[1]["error_type"] == "RuntimeError"


def test_fewshot_all_prompt_matches_reference_prompt_after_official_training():
    reference_root = Path("/home/wyqdf/harness/meta-harness/reference_examples")
    if str(reference_root) not in sys.path:
        sys.path.insert(0, str(reference_root))
    _install_reference_import_stubs()

    from text_classification.agents.fewshot_all import FewShotAll
    from text_classification.data import load_dataset_splits_3way

    rows = load_dataset_splits_3way(
        "Symptom2Disease",
        num_train=3,
        num_val=1,
        num_test=1,
        shuffle_seed=42,
    )
    train_rows, val_rows = rows[0], rows[1]

    reference_prompts = []

    def ref_llm(prompt: str) -> str:
        reference_prompts.append(prompt)
        return '{"final_answer": "diabetes"}'

    reference_memory = FewShotAll(ref_llm)
    reference_memory.learn_from_batch(
        [
            {
                "input": row["input"],
                "prediction": row["target"],
                "ground_truth": row["target"],
                "was_correct": True,
            }
            for row in train_rows
        ]
    )
    reference_memory.predict(val_rows[0]["input"])

    class CaptureLLM(StubLLM):
        def complete(self, messages, **kwargs):
            self.prompt_text = "\n\n".join(message["content"] for message in messages)
            return '{"final_answer": "diabetes"}'

    llm = CaptureLLM()
    ours = load_agent_memory("fewshot_all", lambda prompt: llm.complete([{"role": "user", "content": prompt}]))
    ours.learn_from_batch(
        [
            {
                "input": row["input"],
                "prediction": row["target"],
                "ground_truth": row["target"],
                "was_correct": True,
            }
            for row in train_rows
        ]
    )
    ours.predict(val_rows[0]["input"])

    assert llm.prompt_text == reference_prompts[0]


def test_official_eval_val_saves_memory_and_test_loads_it(tmp_path):
    project = tmp_path
    (project / "agents").mkdir()
    (project / "data/demo").mkdir(parents=True)
    agent_path = project / "agents/last_label.py"
    agent_path.write_text(
        """
import json
from harness.agent_protocol import BaseAgentMemory

class LastLabel(BaseAgentMemory):
    def __init__(self, llm):
        super().__init__(llm)
        self.label = ""
    def predict(self, input):
        prompt = f"Predict: {input}"
        self.call_llm(prompt)
        return self.label, {"full_response": self.label}
    def learn_from_batch(self, batch_results):
        self.label = batch_results[-1]["ground_truth"]
    def get_state(self):
        return json.dumps({"label": self.label})
    def set_state(self, state):
        self.label = json.loads(state)["label"]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    for split, labels in {
        "train": ["B", "B"],
        "val": ["B"],
        "test": ["B"],
    }.items():
        rows = [
            {"id": f"{split}-{idx}", "text": f"{split} text {idx}", "label": label}
            for idx, label in enumerate(labels)
        ]
        (project / f"data/demo/{split}.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )
    dataset = DatasetSpec(
        "demo",
        str(project / "data/demo/train.jsonl"),
        str(project / "data/demo/val.jsonl"),
        str(project / "data/demo/test.jsonl"),
    )
    config = {
        "seed": 42,
        "mode": "offline",
        "batch_size": 1,
        "num_epochs": 1,
        "combined_eval": False,
        "output_root": str(project / "run"),
        "model_short": "stub",
        "project_root": str(project),
        "agent_protocol_only": True,
    }

    val_result = evaluate_system_official(agent_path, [dataset], "val", StubLLM(), config, max_workers=1)
    test_result = evaluate_system_official(agent_path, [dataset], "test", StubLLM(), config, max_workers=1)

    assert val_result.average == 1.0
    assert test_result.average == 1.0
    assert (project / "run/logs/demo/last_label/stub/memory.json").exists()
    assert (project / "run/logs/demo/last_label/stub/val.json").exists()
    assert (project / "run/results/demo/last_label/stub/test.json").exists()


def test_official_eval_matches_reference_inner_loop_on_dual_interface_system(tmp_path):
    reference_root = Path("/home/wyqdf/harness/meta-harness/reference_examples")
    if str(reference_root) not in sys.path:
        sys.path.insert(0, str(reference_root))
    _install_reference_import_stubs()

    from text_classification.data.evaluators import get_evaluator
    from text_classification.inner_loop import evaluate_memory, run_inner_loop
    import importlib.util

    system_path = tmp_path / "dual_memory.py"
    system_path.write_text(
        """
import json
from harness.memory import Prediction

class DualMemory:
    name = "dual_memory"

    def __init__(self):
        self.labels = []
        self.last_prompt = ""

    def learn(self, examples, llm):
        self.labels = [example.label for example in examples]

    def learn_from_batch(self, batch_results):
        self.labels = [row["ground_truth"] for row in batch_results]

    def predict(self, arg, llm=None):
        label = sorted(set(self.labels))[0] if self.labels else "unknown"
        if isinstance(arg, str):
            self.last_prompt = f"Predict: {arg}"
            return label, {"prompt_text": self.last_prompt}
        self.last_prompt = f"Predict: {arg.text}"
        return Prediction(
            label=label,
            raw_output=json.dumps({"final_answer": label}),
            prompt=[{"role": "user", "content": self.last_prompt}],
            retrieved=[],
            metadata={},
        )

    def get_last_prompt_info(self):
        return {
            "prompt_len": len(self.last_prompt),
            "prompt_hash": "deadbeef",
            "prompt_text": self.last_prompt,
        }

    def get_state(self):
        return json.dumps({"labels": self.labels})

    def set_state(self, state):
        self.labels = json.loads(state)["labels"]

def build_memory_system(config=None):
    return DualMemory()
""",
        encoding="utf-8",
    )

    train_rows = [
        {"input": "train 1", "target": "alpha"},
        {"input": "train 2", "target": "beta"},
        {"input": "train 3", "target": "gamma"},
    ]
    val_rows = [
        {"input": "val 1", "target": "gamma"},
        {"input": "val 2", "target": "beta"},
    ]
    test_rows = [
        {"input": "test 1", "target": "gamma"},
        {"input": "test 2", "target": "alpha"},
    ]

    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    test_path = tmp_path / "test.jsonl"
    for path, rows in [(train_path, train_rows), (val_path, val_rows), (test_path, test_rows)]:
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    spec = DatasetSpec("Symptom2Disease", str(train_path), str(val_path), str(test_path))
    evaluator = get_evaluator("Symptom2Disease")

    module_spec = importlib.util.spec_from_file_location("dual_memory", system_path)
    module = importlib.util.module_from_spec(module_spec)
    assert module_spec and module_spec.loader
    module_spec.loader.exec_module(module)
    memory = module.build_memory_system({})

    run_inner_loop(
        memory=memory,
        examples=train_rows,
        check_answer=evaluator,
        batch_size=1,
        max_workers=2,
        logger=None,
        mode="offline",
        num_epochs=1,
        val_examples=None,
        skip_train_eval=True,
    )
    official_eval = evaluate_memory(
        memory,
        val_rows,
        evaluator,
        max_workers=2,
    )

    ours = evaluate_system_official(
        system_path,
        [spec],
        "val",
        StubLLM(),
        {
            "seed": 42,
            "mode": "offline",
            "num_epochs": 1,
            "batch_size": 1,
            "combined_eval": False,
        },
        max_workers=2,
    )

    assert official_eval["accuracy"] == ours.average
    assert [row["prediction"] for row in official_eval["predictions"]] == [
        row["prediction"] for row in ours.traces
    ]
