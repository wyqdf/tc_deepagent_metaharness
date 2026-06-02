"""No-memory baseline for the official-like agent protocol."""
# Baseline: classify each example without storing any memory.


from __future__ import annotations

from typing import Any

from harness.agent_protocol import BaseAgentMemory, extract_json_field


PROMPT = """Answer the following question.

{input}

**Answer in this exact JSON format:**
{{
  "reasoning": "[Your chain of thought / reasoning process]",
  "final_answer": "[Your concise final answer here]"
}}
"""


class NoMemory(BaseAgentMemory):
    def __init__(self, llm):
        super().__init__(llm)
        self._state = "{}"

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        # Ask the solver directly with no stored examples or retrieval state.
        response = self.call_llm(PROMPT.format(input=input))
        return extract_json_field(response, "final_answer"), {"full_response": response}

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        # Keep this baseline stateless.
        return None

    def get_state(self) -> str:
        # Return the tiny serialized placeholder state.
        return self._state

    def set_state(self, state: str) -> None:
        # Restore the placeholder state string unchanged.
        self._state = state
