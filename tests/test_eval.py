"""Tests for benchmark evaluation."""
# Regression checks for the harness contract continue below.


from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from harness.eval import DatasetSpec, BenchmarkResult, evaluate_system, load_jsonl
from harness.llm import StubLLM


def test_load_jsonl_reads_examples(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "1",
                "text": "alpha",
                "label": "A",
                "choices": ["A", "B"],
                "metadata": {"dataset": "demo"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    examples = load_jsonl(path)

    assert examples[0].id == "1"
    assert examples[0].choices == ["A", "B"]


def test_load_jsonl_supports_official_question_target_schema(tmp_path):
    path = tmp_path / "uspto" / "train.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "question": "Context: The reaction type is Protections.\nInput: CC(C)(C)OC(=O)NCCc1ccc(N)cc1\nAnswer: ",
                "target": "CC(C)(C)OC(=O)OC(=O)OC(C)(C)C.NCCc1ccc(N)cc1",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    examples = load_jsonl(path)

    assert examples[0].label.startswith("CC(C)(C)OC(=O)")
    assert "Retrosynthesis Problem:" in examples[0].text
    assert examples[0].choices == []


def test_evaluate_system_runs_train_then_split(tmp_path):
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    test = tmp_path / "test.jsonl"
    train.write_text(
        json.dumps({"id": "t1", "text": "alpha", "label": "A", "choices": ["A", "B"]}) + "\n",
        encoding="utf-8",
    )
    val.write_text(
        json.dumps({"id": "v1", "text": "alpha", "label": "A", "choices": ["A", "B"]}) + "\n",
        encoding="utf-8",
    )
    test.write_text("", encoding="utf-8")
    dataset = DatasetSpec("demo", str(train), str(val), str(test))

    result = evaluate_system("systems/baseline.py", [dataset], "val", StubLLM('{"label": "A"}'), {"k": 1})

    assert isinstance(result, BenchmarkResult)
    assert result.system_name == "lexical_fewshot"
    assert result.per_dataset == {"demo": 1.0}
    assert result.average == 1.0
    assert result.summary["accuracy"] == 1.0
    assert result.summary["correct"] == 1
    assert result.summary["total"] == 1
    assert result.traces[0]["dataset"] == "demo"
    assert result.traces[0]["was_correct"] is True
    assert result.total == 1
    assert result.all_failed is False


def test_evaluate_system_supports_official_symptom_schema(tmp_path):
    train = tmp_path / "symptom_diagnosis" / "train.jsonl"
    val = tmp_path / "symptom_diagnosis" / "val.jsonl"
    test = tmp_path / "symptom_diagnosis" / "test.jsonl"
    train.parent.mkdir(parents=True)
    train.write_text(
        json.dumps({"question": "I feel thirsty and tired.", "answer": "diabetes"}) + "\n",
        encoding="utf-8",
    )
    val.write_text(
        json.dumps({"question": "I have heartburn.", "answer": "gastroesophageal reflux disease"}) + "\n",
        encoding="utf-8",
    )
    test.write_text("", encoding="utf-8")
    dataset = DatasetSpec("Symptom2Disease", str(train), str(val), str(test))

    result = evaluate_system("systems/baseline.py", [dataset], "val", StubLLM('{"final_answer": "gastroesophageal reflux disease"}'), {"k": 1})

    assert result.total == 1
    assert result.traces[0]["dataset"] == "Symptom2Disease"
    assert result.traces[0]["was_correct"] is True


def test_evaluate_system_with_concurrency_keeps_trace_order(tmp_path):
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    test = tmp_path / "test.jsonl"
    train.write_text(
        json.dumps({"id": "t1", "text": "alpha", "label": "A", "choices": ["A", "B"]}) + "\n",
        encoding="utf-8",
    )
    val.write_text(
        "\n".join(
            [
                json.dumps({"id": "v1", "text": "alpha one", "label": "A", "choices": ["A", "B"]}),
                json.dumps({"id": "v2", "text": "alpha two", "label": "A", "choices": ["A", "B"]}),
                json.dumps({"id": "v3", "text": "alpha three", "label": "A", "choices": ["A", "B"]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    test.write_text("", encoding="utf-8")
    dataset = DatasetSpec("demo", str(train), str(val), str(test))

    result = evaluate_system(
        "systems/baseline.py",
        [dataset],
        "val",
        StubLLM('{"final_answer": "A"}'),
        {"k": 1},
        max_workers=3,
    )

    assert [row["id"] for row in result.traces] == ["v1", "v2", "v3"]
    assert result.summary["correct"] == 3


def test_official_mce_loader_matches_reference_first_examples():
    reference_root = Path("/home/wyqdf/harness/meta-harness/reference_examples")
    if str(reference_root) not in sys.path:
        sys.path.insert(0, str(reference_root))
    if "datasets" not in sys.modules:
        stub = types.ModuleType("datasets")
        stub.load_dataset = lambda *args, **kwargs: None
        sys.modules["datasets"] = stub

    from text_classification.data.loaders import load_mce_dataset

    paths = {
        "USPTO": "/home/wyqdf/harness/meta-harness/reference_examples/text_classification/data/uspto/train.jsonl",
        "Symptom2Disease": "/home/wyqdf/harness/meta-harness/reference_examples/text_classification/data/symptom_diagnosis/train.jsonl",
        "LawBench": "/home/wyqdf/harness/meta-harness/reference_examples/text_classification/data/crime_prediction/train.jsonl",
    }
    for task, path in paths.items():
        reference = load_mce_dataset(task, split="train", limit=1)[0]
        ours = load_jsonl(path, task=task)[0]

        assert ours.text == reference["input"]
        assert ours.label == reference["target"]
        assert ours.metadata["raw_question"] == reference["raw_question"]
        assert ours.metadata["context"] == reference["context"]
