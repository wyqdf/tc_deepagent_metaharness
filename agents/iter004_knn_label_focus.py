"""BM25 kNN label voting with focused candidate comparison.

Mechanism change vs iter001_label_coverage_bm25: Fundamentally different PROMPT
ORGANIZATION. Instead of showing examples from many labels with label-coverage,
this system:
1. Uses BM25 to find top-K nearest neighbors
2. Votes on their labels (weighted by BM25 score) to identify top-3 candidates
3. For each candidate label, retrieves the best examples from the full pool
4. Presents examples GROUPED BY CANDIDATE LABEL in a structured comparison

This focuses the model's attention on the 3 most likely labels with more
evidence per candidate (3 examples each), rather than spreading attention
across 10+ labels with 1-2 examples each. The structured comparison format
makes discrimination easier.
"""
# Candidate memory-system logic continues below.


from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from typing import Any

from harness.agent_protocol import BaseAgentMemory, extract_json_field

PROMPT_TEMPLATE = """Solve the problem below. Based on similar cases, the answer is most likely one of the {n} candidates below.

{label_sections}

**Problem:**
{input}

**Instructions:**
- Compare the problem against examples for each candidate answer above.
- The answer is most likely one of: {label_list}
- But if none fits well, you may choose a different answer.
- Respond in JSON format.

{{"reasoning": "[compare against each candidate]", "final_answer": "[your answer]"}}"""

FALLBACK_TEMPLATE = """Solve the problem below based on the examples provided.

{examples_section}

**Problem:**
{input}

**Instructions:**
- Follow the patterns shown in the examples.
- Respond in JSON format.

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""

MAX_CHARS = 30000
VOTE_K = 24
CANDIDATE_LABELS = 3
EXAMPLES_PER_LABEL = 3
CANDIDATE_POOL = 64
NGRAM_MIN = 2
NGRAM_MAX = 3


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


class KnnLabelFocus(BaseAgentMemory):
    """BM25 kNN voting to narrow candidates, then focused comparison."""

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

    def _score_all(self, query: str) -> list[float]:
        qtoks = _tokenize(query)
        return [
            _bm25_score(qtoks, self._doc_tfs[i], self._doc_lens[i], self._avgdl, self._idf)
            for i in range(len(self.examples))
        ]

    def _vote_labels(self, query: str) -> list[str]:
        """Get top candidate labels by score-weighted kNN voting."""
        scores = self._score_all(query)
        n = len(scores)
        top_idxs = sorted(range(n), key=lambda i: -scores[i])[:VOTE_K]
        label_scores: Counter = Counter()
        for i in top_idxs:
            label = self.examples[i].get("target", "")
            label_scores[label] += scores[i]
        return [label for label, _ in label_scores.most_common(CANDIDATE_LABELS)]

    def _get_best_for_label(self, query: str, label: str, k: int) -> list[int]:
        """Get top-k examples for a specific label, ranked by BM25."""
        idxs = self._label_to_idxs.get(label, [])
        if not idxs:
            return []
        qtoks = _tokenize(query)
        scored = [
            (_bm25_score(qtoks, self._doc_tfs[i], self._doc_lens[i], self._avgdl, self._idf), i)
            for i in idxs
        ]
        scored.sort(reverse=True)
        return [i for _, i in scored[:k]]

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        self._ensure_index()
        if not self.examples:
            prompt = FALLBACK_TEMPLATE.format(examples_section="", input=input)
            response = self.call_llm(prompt)
            answer = extract_json_field(response, "final_answer")
            return answer, {"full_response": response}

        # Vote on candidate labels
        top_labels = self._vote_labels(input)
        if not top_labels:
            prompt = FALLBACK_TEMPLATE.format(examples_section="", input=input)
            response = self.call_llm(prompt)
            answer = extract_json_field(response, "final_answer")
            return answer, {"full_response": response}

        # Build structured comparison sections
        sections = []
        total_chars = 0
        for label in top_labels:
            idxs = self._get_best_for_label(input, label, EXAMPLES_PER_LABEL)
            if not idxs:
                continue
            section_parts = [f'**Candidate answer: "{label}"**']
            for i in idxs:
                ex = self.examples[i]
                q = ex.get("raw_question", ex["input"])
                entry = f"  Example: {q}\n  Answer: {ex['target']}"
                if total_chars + len(entry) > MAX_CHARS:
                    break
                section_parts.append(entry)
                total_chars += len(entry) + 2
            sections.append("\n".join(section_parts))

        label_sections = "\n\n".join(sections)
        label_list = ", ".join(f'"{l}"' for l in top_labels)
        prompt = PROMPT_TEMPLATE.format(
            n=len(top_labels),
            label_sections=label_sections,
            input=input,
            label_list=label_list,
        )
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")
        return answer, {
            "full_response": response,
            "candidate_labels": top_labels,
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
