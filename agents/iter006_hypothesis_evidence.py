"""Hypothesis-conditioned evidence comparison.

Mechanism change vs iter005_confusion_verify (base): Instead of a two-pass
predict-then-verify approach with only 2 candidate labels, this uses a
HYPOTHESIS-CONDITIONED approach:

Pass 1: Standard BM25 retrieval + label-coverage selection, but asks the
model to output its top-3 candidate answers (not just 1).

Pass 2: For each of the top-3 candidates that exist in training data,
retrieves the top-2 most relevant examples with that label. Presents a
structured "evidence per hypothesis" prompt and asks the model to compare
the evidence quality and commit to one answer.

Key differences from prior systems:
- iter005_confusion_verify: only compares 2 labels (predicted + alternative)
- iter005_multi_strategy_vote: uses 3 retrieval strategies but same prompt
- iter004_knn_label_focus: narrowed to 3 labels but in a SINGLE pass
  (scored 38.2% because it removed the broad retrieval)

This approach keeps the proven broad retrieval for pass 1 AND adds focused
evidence comparison in pass 2 across 3 hypotheses.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from typing import Any

from harness.agent_protocol import BaseAgentMemory, extract_json_field

STAGE1_PROMPT = """Solve the problem below based on the examples provided.

{examples_section}

**Problem:**
{input}

**Instructions:**
- The examples above are the most relevant prior cases.
- Give your top guess AND up to 2 alternate guesses you'd consider.
- Respond in JSON format.

{{"reasoning": "[brief reasoning]", "candidates": ["best guess", "alternate 1", "alternate 2"], "final_answer": "[best guess]"}}"""

STAGE2_PROMPT = """Solve the problem below. You identified a few plausible answers; for each, supporting examples (where that answer was correct) are listed below. Compare the supporting evidence and commit to ONE final answer.

{evidence_section}

**Problem:**
{input}

**Instructions:**
- For each hypothesis, weigh whether the supporting examples actually fit the problem.
- Pick the hypothesis whose evidence best matches; if none fit, give your own answer.
- Respond in JSON format.

{{"reasoning": "[per-hypothesis assessment]", "final_answer": "[your final answer]"}}"""

MAX_CHARS = 28000
TOP_K = 14
CANDIDATE_POOL = 64
NGRAM_MIN = 2
NGRAM_MAX = 3
ANCHORS = 3
PER_LABEL_CAP = 2
MAX_HYPOTHESES = 3
PER_HYPOTHESIS_K = 2
MIN_EXAMPLES_FOR_STAGE2 = 6


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


def _extract_candidates(response: str) -> list[str]:
    """Parse stage-1 response for the 'candidates' list."""
    cands: list[str] = []
    try:
        data = json.loads(response)
        if isinstance(data, dict) and isinstance(data.get("candidates"), list):
            cands = [str(c).strip() for c in data["candidates"] if str(c).strip()]
    except (json.JSONDecodeError, ValueError):
        pass
    if not cands:
        for match in re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", response):
            try:
                data = json.loads(match.group(1))
                if isinstance(data, dict) and isinstance(data.get("candidates"), list):
                    cands = [str(c).strip() for c in data["candidates"] if str(c).strip()]
                    break
            except (json.JSONDecodeError, ValueError):
                pass
    if not cands:
        # Try to find JSON object in text
        for start in range(len(response)):
            if response[start] != "{":
                continue
            depth = 0
            in_str = False
            esc = False
            for pos in range(start, min(start + 2000, len(response))):
                ch = response[pos]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                elif ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            data = json.loads(response[start:pos+1])
                            if isinstance(data, dict) and isinstance(data.get("candidates"), list):
                                cands = [str(c).strip() for c in data["candidates"] if str(c).strip()]
                        except (json.JSONDecodeError, ValueError):
                            pass
                        break
            if cands:
                break
    return cands[:MAX_HYPOTHESES]


class HypothesisEvidence(BaseAgentMemory):
    """Hypothesis-conditioned evidence comparison."""

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

    def _select(self, query: str) -> list[int]:
        self._ensure_index()
        if not self.examples:
            return []
        qtoks = _tokenize(query)
        n = len(self.examples)
        scores = sorted(
            range(n),
            key=lambda i: -_bm25_score(
                qtoks, self._doc_tfs[i], self._doc_lens[i], self._avgdl, self._idf
            ),
        )
        pool = scores[:CANDIDATE_POOL]
        labels = [self.examples[i].get("target", "") for i in range(n)]
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

    def _get_label_examples(self, query: str, label: str, k: int) -> list[int]:
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

    def _format_example(self, i: int) -> str:
        ex = self.examples[i]
        q = ex.get("raw_question", ex["input"])
        return f"Q: {q}\nA: {ex['target']}"

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        idxs = self._select(input)

        # Build examples section for stage 1
        parts = []
        total = 0
        for i in idxs:
            part = self._format_example(i)
            if total + len(part) > MAX_CHARS:
                break
            parts.append(part)
            total += len(part) + 2
        examples_section = "\n\n".join(parts)

        prompt1 = STAGE1_PROMPT.format(examples_section=examples_section, input=input)
        response1 = self.call_llm(prompt1)
        stage1_answer = extract_json_field(response1, "final_answer")

        # If not enough examples for stage 2, return stage 1 answer
        if len(self.examples) < MIN_EXAMPLES_FOR_STAGE2 or not stage1_answer:
            return stage1_answer, {"full_response": response1, "stage": 1}

        # Extract candidate hypotheses
        candidates = _extract_candidates(response1)
        if not candidates:
            candidates = [stage1_answer]

        # Filter to candidates that have matching examples in training
        valid_candidates = []
        for c in candidates:
            if c in self._label_to_idxs and self._label_to_idxs[c]:
                valid_candidates.append(c)
        # Ensure stage1_answer is included if valid
        if stage1_answer in self._label_to_idxs and stage1_answer not in valid_candidates:
            valid_candidates.insert(0, stage1_answer)

        if len(valid_candidates) < 2:
            # Not enough hypotheses to compare, return stage 1
            return stage1_answer, {"full_response": response1, "stage": 1}

        # Stage 2: build evidence per hypothesis
        evidence_parts = []
        evidence_total = 0
        for hyp in valid_candidates[:MAX_HYPOTHESES]:
            hyp_idxs = self._get_label_examples(input, hyp, PER_HYPOTHESIS_K)
            if not hyp_idxs:
                continue
            block = f'**Hypothesis: "{hyp}"**\nSupporting examples:'
            for i in hyp_idxs:
                ex_text = self._format_example(i)
                block += f"\n{ex_text}"
            if evidence_total + len(block) > MAX_CHARS:
                break
            evidence_parts.append(block)
            evidence_total += len(block) + 2

        if len(evidence_parts) < 2:
            return stage1_answer, {"full_response": response1, "stage": 1}

        evidence_section = "\n\n".join(evidence_parts)
        prompt2 = STAGE2_PROMPT.format(evidence_section=evidence_section, input=input)
        response2 = self.call_llm(prompt2)
        final_answer = extract_json_field(response2, "final_answer")

        return final_answer or stage1_answer, {
            "full_response": response2,
            "stage": 2,
            "stage1_answer": stage1_answer,
            "candidates": valid_candidates[:MAX_HYPOTHESES],
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
