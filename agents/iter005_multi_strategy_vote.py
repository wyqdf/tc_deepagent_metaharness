"""Multi-strategy voting: 3 diverse retrieval strategies, majority vote.

Mechanism change vs iter001_label_coverage_bm25 (base): Instead of a single
retrieval+prediction, uses 3 DIFFERENT retrieval strategies to select examples,
makes 3 independent predictions, and takes majority vote.

The 3 strategies provide genuinely different views of the evidence:
1. Pure BM25 top-k (best lexical matches, may cluster on one label)
2. Label-coverage (ensures diverse labels represented)
3. Recency-weighted BM25 (prefers recent examples which may be more representative)

This is fundamentally different from two-pass verify (predict+verify) or
contrastive retrieval (positive+negative examples). It's an ENSEMBLE approach
where diversity of evidence leads to more robust predictions.

Cost: 3 LLM calls per prediction. Justified if the ensemble reduces variance.
When all 3 agree, confidence is high. When they disagree, majority is more
likely correct than any single strategy.
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
- The examples above are the most relevant prior cases.
- Follow the patterns shown in the examples.
- Respond in JSON format.

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""

MAX_CHARS = 28000
TOP_K = 12
CANDIDATE_POOL = 64
NGRAM_MIN = 2
NGRAM_MAX = 3
ANCHORS = 3
PER_LABEL_CAP = 2
RECENCY_WEIGHT = 0.3


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


class MultiStrategyVote(BaseAgentMemory):
    """3 diverse retrieval strategies with majority vote."""

    def __init__(self, llm):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
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

    def _get_bm25_scores(self, query: str) -> list[float]:
        """Get BM25 scores for all examples."""
        qtoks = _tokenize(query)
        return [
            _bm25_score(qtoks, self._doc_tfs[i], self._doc_lens[i], self._avgdl, self._idf)
            for i in range(len(self.examples))
        ]

    def _strategy_pure_bm25(self, bm25_scores: list[float]) -> list[int]:
        """Strategy 1: Pure top-k by BM25 score."""
        ranked = sorted(range(len(bm25_scores)), key=lambda i: -bm25_scores[i])
        return ranked[:TOP_K]

    def _strategy_label_coverage(self, bm25_scores: list[float]) -> list[int]:
        """Strategy 2: BM25 with label-coverage selection."""
        ranked = sorted(range(len(bm25_scores)), key=lambda i: -bm25_scores[i])
        pool = ranked[:CANDIDATE_POOL]
        labels = [self.examples[i].get("target", "") for i in range(len(self.examples))]
        label_counts: dict[str, int] = defaultdict(int)
        chosen: list[int] = []

        for i in pool[:ANCHORS]:
            chosen.append(i)
            label_counts[labels[i]] += 1
            if len(chosen) >= TOP_K:
                return chosen

        for i in pool:
            if i in chosen:
                continue
            if label_counts[labels[i]] == 0:
                chosen.append(i)
                label_counts[labels[i]] += 1
                if len(chosen) >= TOP_K:
                    return chosen

        for i in pool:
            if i in chosen:
                continue
            if label_counts[labels[i]] < PER_LABEL_CAP:
                chosen.append(i)
                label_counts[labels[i]] += 1
                if len(chosen) >= TOP_K:
                    return chosen

        for i in pool:
            if i in chosen:
                continue
            chosen.append(i)
            if len(chosen) >= TOP_K:
                break
        return chosen

    def _strategy_recency_weighted(self, bm25_scores: list[float]) -> list[int]:
        """Strategy 3: BM25 + recency bonus (prefer recent examples)."""
        n = len(bm25_scores)
        if n == 0:
            return []
        max_score = max(bm25_scores) if bm25_scores else 1.0
        if max_score == 0:
            max_score = 1.0
        combined = [
            bm25_scores[i] + RECENCY_WEIGHT * (i / n) * max_score
            for i in range(n)
        ]
        ranked = sorted(range(n), key=lambda i: -combined[i])
        return ranked[:TOP_K]

    def _format_examples(self, idxs: list[int]) -> str:
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
        self._ensure_index()
        if not self.examples:
            prompt = PROMPT_TEMPLATE.format(examples_section="", input=input)
            response = self.call_llm(prompt)
            answer = extract_json_field(response, "final_answer")
            return answer, {"full_response": response, "strategy": "none"}

        bm25_scores = self._get_bm25_scores(input)

        # If too few examples, just use label-coverage (single call)
        if len(self.examples) < 10:
            idxs = self._strategy_label_coverage(bm25_scores)
            prompt = PROMPT_TEMPLATE.format(
                examples_section=self._format_examples(idxs), input=input
            )
            response = self.call_llm(prompt)
            answer = extract_json_field(response, "final_answer")
            return answer, {"full_response": response, "strategy": "label_coverage_only"}

        # 3 strategies
        strategies = [
            ("pure_bm25", self._strategy_pure_bm25(bm25_scores)),
            ("label_coverage", self._strategy_label_coverage(bm25_scores)),
            ("recency_weighted", self._strategy_recency_weighted(bm25_scores)),
        ]

        predictions = []
        responses = []
        for name, idxs in strategies:
            prompt = PROMPT_TEMPLATE.format(
                examples_section=self._format_examples(idxs), input=input
            )
            response = self.call_llm(prompt)
            answer = extract_json_field(response, "final_answer")
            predictions.append(answer)
            responses.append(response)

        # Majority vote
        vote_counter = Counter(p for p in predictions if p)
        if vote_counter:
            winner = vote_counter.most_common(1)[0][0]
        else:
            winner = predictions[0] if predictions else ""

        return winner, {
            "full_response": responses[-1],
            "predictions": predictions,
            "winner": winner,
            "unanimous": len(set(predictions)) == 1,
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
