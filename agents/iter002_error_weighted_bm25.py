"""Error-weighted BM25 retrieval with hard-label prioritization.

Mechanism change vs iter001_label_coverage_bm25: Changes the LEARNING STRATEGY
and RETRIEVAL RANKING. Tracks which examples were misclassified during training
(via was_correct field in batch_results) and boosts their BM25 retrieval scores
by a multiplier. Also tracks per-label error rates and prioritizes coverage of
hard labels (those with high error rates) over easy ones.

This addresses a different failure mode: the current system treats all examples
equally during retrieval. But examples that were previously misclassified are
more informative — they represent decision boundaries where the model needs
extra guidance. By boosting these examples, the prompt shows the model exactly
the cases it finds hardest.
"""
# Candidate memory-system logic continues below.


from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from typing import Any

from harness.agent_protocol import BaseAgentMemory, extract_json_field

PROMPT_TEMPLATE = """Solve the problem below based on the examples provided.

{examples_section}

**Problem:**
{input}

**Instructions:**
- The examples above include difficult boundary cases. Pay close attention to subtle differences.
- Follow the patterns shown in the examples.
- Respond in JSON format.

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""

MAX_CHARS = 30000
TOP_K = 16
CANDIDATE_POOL = 64
NGRAM_MIN = 2
NGRAM_MAX = 3
ERROR_BOOST = 2.5
HARD_LABEL_SLOTS = 4


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


class ErrorWeightedBm25(BaseAgentMemory):
    """BM25 retrieval with error-weighted scoring and hard-label prioritization."""

    def __init__(self, llm):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self.weights: list[float] = []
        self.label_errors: Counter = Counter()
        self.label_total: Counter = Counter()
        self._docs_tokens: list[list[str]] = []
        self._doc_tfs: list[Counter] = []
        self._doc_lens: list[int] = []
        self._idf: dict[str, float] = {}
        self._avgdl: float = 0.0
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
        self._index_dirty = False

    def _get_hard_labels(self) -> list[str]:
        """Return labels sorted by difficulty (highest error rate first)."""
        difficulties = []
        for label in self.label_total:
            total = self.label_total[label]
            errors = self.label_errors[label]
            rate = errors / max(total, 1)
            if rate > 0:
                difficulties.append((rate, label))
        difficulties.sort(reverse=True)
        return [label for _, label in difficulties]

    def _select(self, query: str) -> list[int]:
        self._ensure_index()
        if not self.examples:
            return []
        qtoks = _tokenize(query)
        n = len(self.examples)

        # Compute weighted BM25 scores
        scored = []
        for i in range(n):
            raw = _bm25_score(
                qtoks, self._doc_tfs[i], self._doc_lens[i], self._avgdl, self._idf
            )
            scored.append((raw * self.weights[i], i))
        scored.sort(reverse=True)
        pool = scored[:CANDIDATE_POOL]

        labels = [self.examples[i].get("target", "") for i in range(n)]
        chosen: list[int] = []
        chosen_labels: set[str] = set()

        # Phase 1: top 3 anchors by weighted score
        for _, i in pool[:3]:
            chosen.append(i)
            chosen_labels.add(labels[i])
            if len(chosen) >= TOP_K:
                return chosen

        # Phase 2: ensure hard labels are represented
        hard_labels = self._get_hard_labels()
        slots_used = 0
        for label in hard_labels:
            if label in chosen_labels:
                continue
            if slots_used >= HARD_LABEL_SLOTS:
                break
            for _, i in pool:
                if i in chosen:
                    continue
                if labels[i] == label:
                    chosen.append(i)
                    chosen_labels.add(label)
                    slots_used += 1
                    break
            if len(chosen) >= TOP_K:
                return chosen

        # Phase 3: label coverage for remaining
        for _, i in pool:
            if i in chosen:
                continue
            if labels[i] not in chosen_labels:
                chosen.append(i)
                chosen_labels.add(labels[i])
                if len(chosen) >= TOP_K:
                    return chosen

        # Phase 4: fill remaining by weighted score
        for _, i in pool:
            if i in chosen:
                continue
            chosen.append(i)
            if len(chosen) >= TOP_K:
                break
        return chosen

    def _format_examples(self, query: str) -> str:
        idxs = self._select(query)
        if not idxs:
            return ""
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
        return "\n\n".join(parts)

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        examples_section = self._format_examples(input)
        prompt = PROMPT_TEMPLATE.format(examples_section=examples_section, input=input)
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")
        return answer, {"full_response": response, "num_examples": len(self.examples)}

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for r in batch_results:
            ex = {"input": str(r["input"]), "target": str(r["ground_truth"])}
            if "raw_question" in r:
                ex["raw_question"] = str(r["raw_question"])
            self.examples.append(ex)

            label = str(r["ground_truth"])
            self.label_total[label] += 1

            # Track errors and boost weight for misclassified examples
            if not r.get("was_correct", True):
                self.weights.append(ERROR_BOOST)
                self.label_errors[label] += 1
            else:
                self.weights.append(1.0)
        self._index_dirty = True

    def get_state(self) -> str:
        return json.dumps(
            {
                "examples": self.examples,
                "weights": self.weights,
                "label_errors": dict(self.label_errors),
                "label_total": dict(self.label_total),
            },
            ensure_ascii=False,
        )

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = data.get("examples", [])
        self.weights = data.get("weights", [1.0] * len(self.examples))
        self.label_errors = Counter(data.get("label_errors", {}))
        self.label_total = Counter(data.get("label_total", {}))
        self._index_dirty = True
