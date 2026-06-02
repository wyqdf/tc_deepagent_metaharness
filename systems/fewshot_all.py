"""Official-style few-shot memory using all training examples as demonstrations."""
# Legacy few-shot baseline retained for older compact checks.


from __future__ import annotations

from typing import Any

from harness.memory import Example, MemorySystem, Prediction, RetrievedExample, extract_label

MAX_CONTEXT_CHARS = 30000
MAX_EXAMPLES = 9999

PROMPT_TEMPLATE = """Solve the problem below based on the examples provided.

{examples_section}

**Problem:**
{input}

**Instructions:**
- Follow the patterns shown in the examples above
- Respond in JSON format

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""


class FewShotAllMemory:
    name = "fewshot_all"

    def __init__(
        self,
        max_context_chars: int = MAX_CONTEXT_CHARS,
        max_examples: int = MAX_EXAMPLES,
    ):
        self.max_context_chars = max_context_chars
        self.max_examples = max_examples
        self.examples: list[Example] = []

    def learn(self, examples: list[Example], llm: Any) -> None:
        # Cache the full training set for few-shot prompting.
        self.examples = list(examples)

    def predict(self, example: Example, llm: Any) -> Prediction:
        # Build one prompt from the cached examples and parse the answer label.
        retrieved = self._format_retrieved_examples(seed=hash(example.text) & 0xFFFFFFFF)
        prompt = self._build_prompt(example, retrieved)
        raw_output = llm.complete(prompt)
        label = extract_label(raw_output, example.choices)
        return Prediction(
            label=label,
            raw_output=raw_output,
            prompt=prompt,
            retrieved=retrieved,
            metadata={"system": self.name, "num_examples": len(retrieved)},
        )

    def _format_retrieved_examples(self, seed: int | None = None) -> list[RetrievedExample]:
        # Shuffle or truncate the cached examples to fit the context budget.
        examples = self.examples[-self.max_examples :]
        if seed is not None:
            import random

            examples = list(examples)
            random.Random(seed).shuffle(examples)

        retrieved: list[RetrievedExample] = []
        total_chars = 0
        for idx, item in enumerate(examples, 1):
            question = item.text
            block = f"Q: {question}\nA: {item.label}"
            if total_chars + len(block) > self.max_context_chars:
                break
            retrieved.append(
                RetrievedExample(
                    id=item.id,
                    text=question,
                    label=item.label,
                    score=float(len(examples) - idx + 1),
                    metadata=item.metadata,
                )
            )
            total_chars += len(block) + 2
        return retrieved

    def _build_prompt(
        self,
        example: Example,
        retrieved: list[RetrievedExample],
    ) -> list[dict[str, str]]:
        # Render the few-shot examples and the query into the final chat prompt.
        examples_section = "\n\n".join(
            f"Q: {item.text}\nA: {item.label}" for item in retrieved
        )
        user = PROMPT_TEMPLATE.format(examples_section=examples_section, input=example.text)
        return [
            {"role": "user", "content": user},
        ]


def build_memory_system(config: dict[str, Any] | None = None) -> MemorySystem:
    # Create the legacy few-shot memory system from optional config settings.
    config = config or {}
    return FewShotAllMemory(
        max_context_chars=int(config.get("max_context_chars", MAX_CONTEXT_CHARS)),
        max_examples=int(config.get("max_examples", MAX_EXAMPLES)),
    )
