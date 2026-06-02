"""Few-shot-all baseline for the official-like agent protocol."""
# Baseline: keep all examples and sample them into the prompt.


from __future__ import annotations

import json
import random
from typing import Any

from harness.agent_protocol import BaseAgentMemory, extract_json_field


PROMPT_TEMPLATE = """Solve the problem below based on the examples provided.

{examples_section}

**Problem:**
{input}

**Instructions:**
- Follow the patterns shown in the examples above
- Respond in JSON format

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""

MAX_CHARS = 30000


class FewShotAll(BaseAgentMemory):
    def __init__(self, llm, max_examples: int = 9999):
        super().__init__(llm)
        self.max_examples = max_examples
        self.examples: list[dict[str, str]] = []

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        # Sample the stored examples into one prompt and ask the solver once.
        seed = hash(input) & 0xFFFFFFFF
        examples_section = self._format_examples_section(seed=seed)
        prompt = PROMPT_TEMPLATE.format(examples_section=examples_section, input=input)
        response = self.call_llm(prompt)
        return extract_json_field(response, "final_answer"), {
            "full_response": response,
            "num_examples": len(self.examples),
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        # Append each training example so later prompts can use all prior data.
        for row in batch_results:
            example = {"input": str(row["input"]), "target": str(row["ground_truth"])}
            if "raw_question" in row:
                example["raw_question"] = str(row["raw_question"])
            self.examples.append(example)

    def get_context_length(self) -> int:
        # Measure how much prompt space the stored examples currently consume.
        return len(self._format_examples_section())

    def get_state(self) -> str:
        # Serialize the full example cache for reuse across runs.
        return json.dumps({"examples": self.examples}, indent=2, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        # Restore the example cache from saved JSON state.
        data = json.loads(state)
        self.examples = [
            {str(key): str(value) for key, value in row.items()}
            for row in data.get("examples", [])
            if isinstance(row, dict)
        ]

    def _format_examples_section(self, seed: int | None = None) -> str:
        # Turn the stored examples into a prompt block, capped by example and char limits.
        if not self.examples:
            return ""
        if seed is not None and len(self.examples) > self.max_examples:
            selected = random.Random(seed).sample(self.examples, self.max_examples)
        else:
            selected = list(self.examples[-self.max_examples :])
            if seed is not None:
                random.Random(seed).shuffle(selected)

        parts: list[str] = []
        total_chars = 0
        for row in selected:
            question = row.get("raw_question", row["input"])
            part = f"Q: {question}\nA: {row['target']}"
            if total_chars + len(part) > MAX_CHARS:
                break
            parts.append(part)
            total_chars += len(part) + 2
        return "\n\n".join(parts)
