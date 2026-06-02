"""Interface skeleton smoke tests."""
# Regression checks for the harness contract continue below.


from __future__ import annotations

import inspect

from harness import eval as eval_module
from harness import llm, loop, memory, metrics, prompts, proposer, store
from systems import baseline


def test_core_interfaces_are_defined():
    assert inspect.isclass(memory.Example)
    assert inspect.isclass(memory.RetrievedExample)
    assert inspect.isclass(memory.Prediction)
    assert hasattr(memory.MemorySystem, "learn")
    assert hasattr(memory.MemorySystem, "predict")

    assert inspect.isclass(llm.LLMConfig)
    assert inspect.isclass(llm.StubLLM)
    assert inspect.isclass(llm.OpenAICompatibleClient)

    assert inspect.isclass(eval_module.DatasetSpec)
    assert inspect.isclass(eval_module.BenchmarkResult)
    assert inspect.isclass(proposer.CandidateProposal)
    assert inspect.isclass(proposer.DeepAgentProposer)
    assert inspect.isclass(store.RunStore)
    assert inspect.isclass(loop.HarnessConfig)
    assert inspect.isclass(baseline.LexicalFewShotMemory)


def test_expected_public_functions_exist():
    for fn in [
        memory.tokenize,
        memory.format_choices,
        memory.safe_json_loads,
        memory.extract_label,
        metrics.normalize_label,
        metrics.accuracy,
        metrics.per_dataset_scores,
        metrics.macro_average,
        eval_module.load_jsonl,
        eval_module.import_memory_system,
        eval_module.evaluate_system,
        prompts.build_classifier_prompt,
        prompts.build_verification_prompt,
        prompts.build_proposer_prompt,
        loop.load_config,
        loop.run_harness,
        baseline.build_memory_system,
    ]:
        assert callable(fn)
