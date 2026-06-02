"""Comparative label-group prediction (single-pass).

Mechanism change vs iter012_mmr_confusion_verify: Replaces the two-pass
(predict + verify) approach with a single-pass comparative format. Instead of
showing flat diverse examples and asking for a prediction, this system:

1. Retrieves a large BM25 candidate pool
2. Identifies top-N candidate labels by vote-counting in the pool
3. For each candidate label, retrieves the top-K most relevant examples
4. Formats the prompt as a GROUPED COMPARISON: examples organized by candidate label
5. Asks the model to compare across groups and pick the best match

Key differences from existing systems:
- iter002_label_grouped_prompt: Grouped ALL labels (too many groups, diluted).
  This only shows top-4 candidate labels = focused comparison.
- iter012_narrowing_verify: Used narrowing + 2 passes. This is single-pass
  with the comparative format as the PRIMARY prediction mechanism.
- iter012_mmr_confusion_verify: Flat examples + binary verify. This uses
  grouped examples with no verify step (saves 1 LLM call).

The hypothesis: A focused comparison among top candidates is more effective
than flat retrieval because it forces the model to explicitly discriminate
between the most likely options. On classification tasks (S2D, LawBench),
this should help. On unique-target tasks (USPTO), falls back to standard
BM25 retrieval since no label repeats.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from typing import Any

from harness.agent_protocol import BaseAgentMemory, extract_json_field

COMPARATIVE_TEMPLATE = """Solve the problem below by comparing it against the candidate answer groups.

{groups_section}

**Problem:**
{input}

**Instructions:**
- Each group above shows examples for one candidate answer.
- Compare the problem against each group and pick the answer whose examples best match.
- The correct answer is most likely one of the candidates shown above.
- Respond in JSON format.

{{"reasoning": "[compare problem against each candidate group]", "final_answer": "[your answer]"}}"""

FLAT_TEMPLATE = """Solve the problem below based on the examples provided.

{examples_section}

**Problem:**
{input}

**Instructions:**
- The examples above span the most plausible answers. Compare the problem to each example and pick the answer whose example best matches.
- Follow the patterns shown in the examples.
- Respond in JSON format.

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""

MAX_CHARS = 28000
CANDIDATE_POOL = 64
NGRAM_MIN = 2
NGRAM_MAX = 3
MAX_CANDIDATE_LABELS = 4
EXAMPLES_PER_LABEL = 3
FLAT_TOP_K = 14
MIN_LABEL_VOTES = 2  # minimum votes to be a candidate label
UNIQUE_LABEL_RATIO = 0.8  # if >80% of labels are unique, use flat retrieval


def _tokenize(s: str) -> list[str]:
    lower = s.lower()
    words = re.findall(r"[a-z0-9\u4e00-\u9fff]+|[\(\)=#\[\]/\\@\+\-\.]", lower)
    compact = re.sub(r"\s+", "", lower)
    ngrams: list[str] = []
    for n in range(NGRAM_MIN, NGRAM_MAX + 1):
        if len(compact) < n:
            continue
        for i in range(len(compact) - n + 1):
            ngrams.append(compact[i : i + n])
    return words + ngrams


def _bm25_idf(docs_tokens: list[list[str]]) -> dict[str, float]:
    n = len(docs_tokens)
    df: Counter = Counter()
    for d in docs_tokens:
        for t in set(d):
            df[t] += 1
    return {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}


def _bm25_score(qtoks, tf, dl, avgdl, idf, k1=1.5, b=0.75):
    s = 0.0
    for t in qtoks:
        f = tf.get(t, 0)
        if not f:
            continue
        denom = f + k1 * (1 - b + b * dl / max(1.0, avgdl))
        s += idf.get(t, 0.0) * f * (k1 + 1) / denom
    return s


class ComparativeLabelGroup(BaseAgentMemory):
    """Single-pass comparative label-group prediction."""

    def __init__(self, llm):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self._docs_tokens: list[list[str]] = []
        self._doc_tfs: list[Counter] = []
        self._doc_lens: list[int] = []
        self._idf: dict[str, float] = {}
        self._avgdl: float = 0.0
        self._label_to_idxs: dict[str, list[int]] = defaultdict(list)
        self._index_dirty = True

    def _ensure_index(self) -> None:
        if not self._index_dirty:
            return
        questions = [ex.get("raw_question") or ex["input"] for ex in self.examples]
        self._docs_tokens = [_tokenize(q) for q in questions]
        self._doc_tfs = [Counter(t) for t in self._docs_tokens]
        self._doc_lens = [len(t) for t in self._docs_tokens]
        n = len(self._docs_tokens)
        self._avgdl = (sum(self._doc_lens) / n) if n else 0.0
        self._idf = _bm25_idf(self._docs_tokens)
        self._label_to_idxs = defaultdict(list)
        for i, ex in enumerate(self.examples):
            self._label_to_idxs[ex.get("target", "")].append(i)
        self._index_dirty = False

    def _is_unique_target_task(self) -> bool:
        """Detect if this is a unique-target task (like USPTO)."""
        if not self.examples:
            return False
        n_labels = len(self._label_to_idxs)
        n_examples = len(self.examples)
        return n_labels / max(1, n_examples) > UNIQUE_LABEL_RATIO

    def _flat_retrieval(self, query: str) -> list[int]:
        """Standard BM25 top-K retrieval for unique-target tasks."""
        self._ensure_index()
        if not self.examples:
            return []
        qtoks = _tokenize(query)
        scores = [
            _bm25_score(qtoks, self._doc_tfs[i], self._doc_lens[i], self._avgdl, self._idf)
            for i in range(len(self.examples))
        ]
        ranked = sorted(range(len(self.examples)), key=lambda i: -scores[i])
        return ranked[:FLAT_TOP_K]

    def _get_candidate_labels(self, query: str) -> list[str]:
        """Get top candidate labels by BM25 vote-counting."""
        self._ensure_index()
        qtoks = _tokenize(query)
        scores = [
            _bm25_score(qtoks, self._doc_tfs[i], self._doc_lens[i], self._avgdl, self._idf)
            for i in range(len(self.examples))
        ]
        pool = sorted(range(len(self.examples)), key=lambda i: -scores[i])[:CANDIDATE_POOL]

        label_votes: Counter = Counter()
        for idx in pool:
            label_votes[self.examples[idx].get("target", "")] += 1

        candidates = []
        for label, votes in label_votes.most_common(MAX_CANDIDATE_LABELS):
            if votes >= MIN_LABEL_VOTES:
                candidates.append(label)
        return candidates

    def _get_label_examples(self, query: str, label: str, k: int) -> list[int]:
        """Get top-k most relevant examples for a specific label."""
        self._ensure_index()
        idxs = self._label_to_idxs.get(label, [])
        if not idxs:
            return []
        qtoks = _tokenize(query)
        scored = []
        for i in idxs:
            s = _bm25_score(qtoks, self._doc_tfs[i], self._doc_lens[i], self._avgdl, self._idf)
            scored.append((s, i))
        scored.sort(reverse=True)
        return [i for _, i in scored[:k]]

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        self._ensure_index()

        if not self.examples:
            prompt = FLAT_TEMPLATE.format(examples_section="", input=input)
            response = self.call_llm(prompt)
            answer = extract_json_field(response, "final_answer")
            return answer, {"full_response": response, "mode": "empty"}

        # For unique-target tasks, use flat retrieval
        if self._is_unique_target_task():
            idxs = self._flat_retrieval(input)
            parts = []
            total = 0
            for i in idxs:
                ex = self.examples[i]
                q = ex.get("raw_question", ex["input"])
                part = f"Q: {q}\nA: {ex['target']}"
                if total + len(part) > MAX_CHARS:
                    break
                parts.append(part)
                total += len(part) + 2
            prompt = FLAT_TEMPLATE.format(examples_section="\n\n".join(parts), input=input)
            response = self.call_llm(prompt)
            answer = extract_json_field(response, "final_answer")
            return answer, {"full_response": response, "mode": "flat"}

        # For classification tasks, use comparative label-group format
        candidate_labels = self._get_candidate_labels(input)

        if len(candidate_labels) < 2:
            # Not enough candidates, fall back to flat
            idxs = self._flat_retrieval(input)
            parts = []
            total = 0
            for i in idxs:
                ex = self.examples[i]
                q = ex.get("raw_question", ex["input"])
                part = f"Q: {q}\nA: {ex['target']}"
                if total + len(part) > MAX_CHARS:
                    break
                parts.append(part)
                total += len(part) + 2
            prompt = FLAT_TEMPLATE.format(examples_section="\n\n".join(parts), input=input)
            response = self.call_llm(prompt)
            answer = extract_json_field(response, "final_answer")
            return answer, {"full_response": response, "mode": "flat_fallback"}

        # Build grouped comparison
        per_group_budget = MAX_CHARS // len(candidate_labels)
        groups_parts = []
        for label in candidate_labels:
            label_idxs = self._get_label_examples(input, label, EXAMPLES_PER_LABEL)
            if not label_idxs:
                continue
            ex_parts = []
            total = 0
            for i in label_idxs:
                ex = self.examples[i]
                q = ex.get("raw_question", ex["input"])
                part = f"  Q: {q}\n  A: {ex['target']}"
                if total + len(part) > per_group_budget:
                    break
                ex_parts.append(part)
                total += len(part) + 2
            if ex_parts:
                groups_parts.append(f'**Candidate answer: "{label}"**\n' + "\n\n".join(ex_parts))

        if not groups_parts:
            prompt = FLAT_TEMPLATE.format(examples_section="", input=input)
            response = self.call_llm(prompt)
            answer = extract_json_field(response, "final_answer")
            return answer, {"full_response": response, "mode": "empty_groups"}

        groups_section = "\n\n---\n\n".join(groups_parts)
        prompt = COMPARATIVE_TEMPLATE.format(groups_section=groups_section, input=input)
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")
        return answer, {
            "full_response": response,
            "mode": "comparative",
            "candidate_labels": candidate_labels,
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for r in batch_results:
            ex = {"input": str(r["input"]), "target": str(r["ground_truth"])}
            if "raw_question" in r:
                ex["raw_question"] = str(r["raw_question"])
            self.examples.append(ex)
        self._index_dirty = True

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = data.get("examples", [])
        self._index_dirty = True
