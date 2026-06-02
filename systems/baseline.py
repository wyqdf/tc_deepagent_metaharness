"""Baseline lexical few-shot memory system."""
# Legacy lexical baseline kept for compatibility tests.


from __future__ import annotations

from typing import Any

from harness.memory import (
    Example,
    MemorySystem,
    Prediction,
    RetrievedExample,
    extract_label,
    format_choices,
    tokenize,
)


class LexicalFewShotMemory:
    name = "lexical_fewshot"

    def __init__(self, k: int = 3):
        self.k = k
        self.examples: list[Example] = []

    def learn(self, examples: list[Example], llm: Any) -> None:
        # Store the full training set for lexical retrieval at prediction time.
        self.examples = list(examples)

    def predict(self, example: Example, llm: Any) -> Prediction:
        # Retrieve the closest examples, build the prompt, and parse the label.
        retrieved = self._retrieve(example)
        prompt = self._build_prompt(example, retrieved)
        raw_output = llm.complete(prompt)
        label = extract_label(raw_output, example.choices)
        return Prediction(
            label=label,
            raw_output=raw_output,
            prompt=prompt,
            retrieved=retrieved,
            metadata={"system": self.name},
        )

    def _retrieve(self, example: Example) -> list[RetrievedExample]:
        # Rank training examples by token overlap with the query text.
        query_tokens = set(tokenize(example.text))
        scored: list[RetrievedExample] = []
        for train_example in self.examples:
            train_tokens = set(tokenize(train_example.text))
            score = float(len(query_tokens & train_tokens))
            scored.append(
                RetrievedExample(
                    id=train_example.id,
                    text=train_example.text,
                    label=train_example.label,
                    score=score,
                    metadata=train_example.metadata,
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[: self.k]

    def _build_prompt(
        self,
        example: Example,
        retrieved: list[RetrievedExample],
    ) -> list[dict[str, str]]:
        # Format the retrieved examples into a few-shot classifier prompt.
        fewshot = "\n\n".join(
            f"Example {idx + 1}\nText: {item.text}\nLabel: {item.label}"
            for idx, item in enumerate(retrieved)
        )
        user = (
            "Classify the text into exactly one of the allowed labels.\n"
            f"Choices: {format_choices(example.choices)}\n\n"
        )
        if fewshot:
            user += f"Relevant training examples:\n{fewshot}\n\n"
        user += f"Text: {example.text}\nReturn JSON: {{\"final_answer\": \"...\"}}"
        return [
            {"role": "system", "content": "You are a careful text classifier."},
            {"role": "user", "content": user},
        ]


def build_memory_system(config: dict[str, Any] | None = None) -> MemorySystem:
    # Create the baseline memory system from optional config settings.
    config = config or {}
    return LexicalFewShotMemory(k=int(config.get("k", 3)))
