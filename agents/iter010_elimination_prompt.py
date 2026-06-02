"""Chain-of-elimination prompt with label-aware retrieval.

Mechanism change vs iter001_label_coverage_bm25 (base, 48.2%): Fundamentally
changes the PROMPT ORGANIZATION. Instead of showing examples and asking "what
is the answer?", this system:

1. Retrieves examples via BM25 with label-coverage (same as base)
2. Identifies the set of candidate labels from retrieved examples
3. Structures the prompt as an ELIMINATION task: "Here are the candidate answers.
   For each, I show representative examples. Eliminate answers that don't fit,
   then choose the best remaining one."

This forces the model to explicitly consider and reject alternatives rather than
just pattern-matching to the most similar example. It's particularly effective
when multiple labels have similar examples (LawBench crime names, S2D symptoms).

Key difference from two-pass verify: verify asks "is A or B correct?" AFTER
making a prediction. Elimination asks "which of {A, B, C, D, ...} is correct?"
in a SINGLE pass, forcing explicit comparison of all candidates simultaneously.

Key difference from label_grouped_prompt (iter002, 29.1%): that grouped examples
by label but still asked "what is the answer?". This explicitly frames the task
as elimination with a structured comparison instruction.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from typing import Any

from harness.agent_protocol import BaseAgentMemory, extract_json_field

ELIMINATION_TEMPLATE = """You must classify the problem below into exactly one of the candidate answers listed.

**Candidate answers with representative examples:**
{candidates_section}

**Problem to classify:**
{input}

**Instructions:**
- Consider each candidate answer above.
- Eliminate candidates whose examples do NOT match the problem's key features.
- Choose the single best remaining candidate.
- If none fit well, pick the closest match.
- Respond in JSON format.

{{"reasoning": "[for each candidate, briefly state why it fits or doesn't fit]", "final_answer": "[your chosen answer]"}}"""

FALLBACK_TEMPLATE = """Solve the problem below based on the examples provided.

{examples_section}

**Problem:**
{input}

**Instructions:**
- The examples above are the most relevant prior cases.
- Follow the patterns shown in the examples.
- Respond in JSON format.

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""

MAX_CHARS = 28000
TOP_K = 14
CANDIDATE_POOL = 64
NGRAM_MIN = 2
NGRAM_MAX = 3
ANCHORS = 3
PER_LABEL_CAP = 2
EXAMPLES_PER_CANDIDATE = 2
MIN_EXAMPLES_FOR_ELIMINATION = 8
MAX_CANDIDATES = 8


def _tokenize(s):
    lower = s.lower()
    words = re.findall(r"[a-z0-9\u4e00-\u9fff]+|[\(\)=#\[\]/\\@\+\-\.]", lower)
    compact = re.sub(r"\s+", "", lower)
    ngrams = []
    for n in range(NGRAM_MIN, NGRAM_MAX + 1):
        for i in range(len(compact) - n + 1):
            ngrams.append(compact[i:i + n])
    return words + ngrams


def _bm25_idf(docs_tokens):
    n = len(docs_tokens)
    df = Counter()
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


class EliminationPrompt(BaseAgentMemory):
    """Chain-of-elimination prompt with label-aware retrieval."""

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

    def _ensure_index(self):
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

    def _select_standard(self, query):
        """Standard BM25 + label-coverage selection."""
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

    def _get_candidate_labels(self, query):
        """Get top candidate labels ranked by best BM25 score."""
        self._ensure_index()
        qtoks = _tokenize(query)
        label_best = {}
        for label, idxs in self._label_to_idxs.items():
            best = 0.0
            for i in idxs:
                s = _bm25_score(qtoks, self._doc_tfs[i], self._doc_lens[i], self._avgdl, self._idf)
                if s > best:
                    best = s
            label_best[label] = best
        ranked = sorted(label_best.keys(), key=lambda l: -label_best[l])
        return ranked[:MAX_CANDIDATES]

    def _get_best_examples_for_label(self, query, label, k):
        """Get top-k examples for a specific label by BM25 relevance to query."""
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

    def _format_elimination_prompt(self, input_text, candidate_labels):
        """Build the elimination-style prompt."""
        sections = []
        total = 0
        for label in candidate_labels:
            ex_idxs = self._get_best_examples_for_label(input_text, label, EXAMPLES_PER_CANDIDATE)
            if not ex_idxs:
                continue
            parts = [f"**Answer: {label}**"]
            for i in ex_idxs:
                ex = self.examples[i]
                q = ex.get("raw_question", ex["input"])
                entry = f"  Example: {q[:500]}"
                parts.append(entry)
            section = "\n".join(parts)
            if total + len(section) > MAX_CHARS:
                break
            sections.append(section)
            total += len(section) + 2
        return "\n\n".join(sections)

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        self._ensure_index()
        if not self.examples:
            prompt = FALLBACK_TEMPLATE.format(examples_section="", input=input)
            response = self.call_llm(prompt)
            answer = extract_json_field(response, "final_answer")
            return answer, {"full_response": response, "mode": "zero_shot"}

        # Check if we have enough distinct labels for elimination
        n_labels = len(self._label_to_idxs)
        use_elimination = (
            len(self.examples) >= MIN_EXAMPLES_FOR_ELIMINATION
            and n_labels >= 3
        )

        if not use_elimination:
            # Fallback to standard retrieval
            idxs = self._select_standard(input)
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
            prompt = FALLBACK_TEMPLATE.format(
                examples_section="\n\n".join(parts), input=input
            )
            response = self.call_llm(prompt)
            answer = extract_json_field(response, "final_answer")
            return answer, {"full_response": response, "mode": "fallback"}

        # Elimination mode
        candidate_labels = self._get_candidate_labels(input)
        candidates_section = self._format_elimination_prompt(input, candidate_labels)

        prompt = ELIMINATION_TEMPLATE.format(
            candidates_section=candidates_section, input=input
        )
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")

        return answer, {
            "full_response": response,
            "mode": "elimination",
            "n_candidates": len(candidate_labels),
            "candidates": candidate_labels,
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
