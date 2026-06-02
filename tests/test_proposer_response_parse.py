"""Tests for parsing real proposer responses without calling DeepAgent."""
# Regression checks for the harness contract continue below.


from __future__ import annotations

from harness.proposer import DeepAgentProposer


def test_proposal_from_response_extracts_manifest_and_python(tmp_path):
    proposer = DeepAgentProposer("claude-opus-4-6", tmp_path, dry_run=False)
    response = """
```json
{"name": "candidate_x", "hypothesis": "better retrieval"}
```

```python
from harness.agent_protocol import BaseAgentMemory

class CandidateX(BaseAgentMemory):
    def predict(self, input): return "x", {}
    def learn_from_batch(self, batch_results): pass
    def get_state(self): return "{}"
    def set_state(self, state): pass
```
"""

    proposal = proposer._proposal_from_response(response, iteration=3)

    assert proposal.name == "candidate_x"
    assert proposal.hypothesis == "better retrieval"
    assert "BaseAgentMemory" in proposal.source_code
