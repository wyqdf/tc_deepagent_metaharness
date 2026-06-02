"""Focused narrowing retrieval with confusion-guided verification.

Mechanism change vs iter005_confusion_verify (base): Fundamentally different
RETRIEVAL strategy for pass1. Instead of selecting TOP_K examples across all
labels (which gives ~1 example per label when there are many labels), this
system first identifies the TOP candidate labels by BM25 voting, then
retrieves MORE examples per candidate label.

The narrowing process:
1. BM25-score all training examples against the query
2. From the top-30 pool, count which labels appear most frequently
3. Select the top-4 candidate labels (most votes in the pool)
4. For each candidate label, retrieve the top-3 most BM25-relevant examples
5. Present examples organized by candidate label in the pass1 prompt

Why this is fundamentally new:
- iter005: Shows 14 examples from any labels, with label-coverage ensuring
  diversity. Each label gets ~1-2 examples. The model must reason with thin
  evidence per candidate.
- iter002_label_grouped_prompt: Grouped examples by label but didn't narrow
  which labels to show. It showed ALL labels from training, which was too many.
- iter004_knn_label_focus: Narrowed to top-3 labels using kNN voting, but used
  a structured comparison format that confused the model. We keep the proven
  flat prompt format.
- This system: Narrows to top-4 labels FIRST (data-driven selection), then
  shows 3 examples per label = 12 examples total. The model gets DEEPER
  evidence for the most likely candidates.

On LawBench (50+ crime types), current approach shows 14 examples from 14
different crimes, giving the model no depth. Narrowing to 4 candidate crimes
with 3 examples each gives the model real patterns to match.

On S2D (24 diseases), narrowing to 4 candidates with 3 examples each should
still capture the correct disease in the candidate set (since BM25 retrieval
already works well here).

On USPTO (unique targets), each label has only 1 example in training, so
narrowing to 4 labels = 4 examples. The system falls back to showing all
available examples from the pool, behaving like base BM25 retrieval.

The verify pass uses confusion-guided alternative selection (proven best).
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from typing import Any

from harness.agent_protocol import BaseAgentMemory, extract_json_field

PASS1_TEMPLATE = """Solve the problem below based on the examples provided.

The examples below are organized by the most likely candidate answers based on similarity to the problem.

{examples_section}

**Problem:**
{input}

**Instructions:**
- The candidate answers shown above are the most likely based on prior cases. Pick the one that best matches the problem.
- Compare the problem to examples in each category and choose the answer with the strongest match.
- Respond in JSON format.

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""

PASS1_FALLBACK = """Solve the problem below based on the examples provided.

{examples_section}

**Problem:**
{input}

**Instructions:**
- The examples above span the most plausible answers. Compare the problem to each example and pick the answer whose example best matches.
- Follow the patterns shown in the examples.
- Respond in JSON format.

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""

VERIFY_TEMPLATE = """You previously predicted the answer below. Now verify it by comparing against focused examples for two candidate answers.

**Your initial prediction:** {prediction}

**Examples for "{label_a}":**
{examples_a}

**Examples for "{label_b}":**
{examples_b}

**Problem:**
{input}

**Instructions:**
- Compare the problem carefully against both sets of examples.
- The correct answer is one of: "{label_a}" or "{label_b}" (or something else if neither fits).
- If your initial prediction was wrong, correct it now.
- Respond in JSON format.

{{"reasoning": "[compare against both candidate labels]", "final_answer": "[your final answer]"}}"""

MAX_CHARS = 28000
NARROW_POOL = 30
NARROW_TOP_LABELS = 4
EXAMPLES_PER_LABEL = 3
MIN_LABELS_FOR_NARROWING = 3
CANDIDATE_POOL = 64
NGRAM_MIN = 2
NGRAM_MAX = 3
VERIFY_EXAMPLES_PER_LABEL = 3
MIN_EXAMPLES_FOR_VERIFY = 6
CONFUSION_MIN = 1
FALLBACK_TOP_K = 14
ANCHORS = 3
PER_LABEL_CAP = 2


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


class NarrowingVerify(BaseAgentMemory):
    """Focused narrowing retrieval + confusion-guided verification."""

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
        self._confusion: dict[str, Counter] = defaultdict(Counter)

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

    def _should_narrow(self) -> bool:
        """Narrowing only makes sense when there are enough labels with multiple examples."""
        multi_example_labels = sum(1 for idxs in self._label_to_idxs.values() if len(idxs) >= 2)
        return multi_example_labels >= MIN_LABELS_FOR_NARROWING

    def _narrowing_select(self, query: str) -> tuple[list[int], list[str]]:
        """Select examples by narrowing to top candidate labels first."""
        self._ensure_index()
        if not self.examples:
            return [], []

        qtoks = _tokenize(query)
        n = len(self.examples)

        # Score all docs
        scores = [
            (i, _bm25_score(qtoks, self._doc_tfs[i], self._doc_lens[i], self._avgdl, self._idf))
            for i in range(n)
        ]
        scores.sort(key=lambda x: -x[1])

        # From top NARROW_POOL, vote on labels
        pool = scores[:NARROW_POOL]
        label_votes: Counter = Counter()
        label_max_score: dict[str, float] = {}
        for idx, score in pool:
            label = self.examples[idx].get("target", "")
            label_votes[label] += 1
            if label not in label_max_score or score > label_max_score[label]:
                label_max_score[label] = score

        # Select top-N candidate labels (by vote count, break ties by max BM25 score)
        candidate_labels = sorted(
            label_votes.keys(),
            key=lambda l: (label_votes[l], label_max_score.get(l, 0)),
            reverse=True,
        )[:NARROW_TOP_LABELS]

        # For each candidate label, get top EXAMPLES_PER_LABEL by BM25
        chosen: list[int] = []
        for label in candidate_labels:
            idxs = self._label_to_idxs.get(label, [])
            if not idxs:
                continue
            label_scored = [
                (_bm25_score(qtoks, self._doc_tfs[i], self._doc_lens[i], self._avgdl, self._idf), i)
                for i in idxs
            ]
            label_scored.sort(reverse=True)
            for _, idx in label_scored[:EXAMPLES_PER_LABEL]:
                if idx not in chosen:
                    chosen.append(idx)

        return chosen, candidate_labels

    def _fallback_select(self, query: str) -> list[int]:
        """Fallback: standard label-coverage selection (same as iter005)."""
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
            if len(chosen) >= FALLBACK_TOP_K:
                return chosen

        for i in pool:
            if i in chosen:
                continue
            if label_counts[labels[i]] == 0:
                chosen.append(i)
                label_counts[labels[i]] += 1
                if len(chosen) >= FALLBACK_TOP_K:
                    return chosen

        for i in pool:
            if i in chosen:
                continue
            if label_counts[labels[i]] < PER_LABEL_CAP:
                chosen.append(i)
                label_counts[labels[i]] += 1
                if len(chosen) >= FALLBACK_TOP_K:
                    return chosen

        for i in pool:
            if i in chosen:
                continue
            chosen.append(i)
            if len(chosen) >= FALLBACK_TOP_K:
                break
        return chosen

    def _find_alternative_label(self, query: str, predicted: str) -> str | None:
        """Find alternative: confusion-guided first, BM25 fallback."""
        if predicted in self._confusion:
            candidates = self._confusion[predicted].most_common()
            for label, count in candidates:
                if count >= CONFUSION_MIN and label in self._label_to_idxs:
                    return label

        self._ensure_index()
        qtoks = _tokenize(query)
        label_best: dict[str, float] = {}
        for i in range(len(self.examples)):
            label = self.examples[i].get("target", "")
            if label == predicted:
                continue
            s = _bm25_score(
                qtoks, self._doc_tfs[i], self._doc_lens[i], self._avgdl, self._idf
            )
            if label not in label_best or s > label_best[label]:
                label_best[label] = s
        if not label_best:
            return None
        return max(label_best, key=label_best.get)

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

    def _format_examples_list(self, idxs: list[int]) -> str:
        parts = []
        total = 0
        for i in idxs:
            ex = self.examples[i]
            q = ex.get("raw_question", ex["input"])
            part = f"Q: {q}\nA: {ex['target']}"
            if total + len(part) > MAX_CHARS // 2:
                break
            parts.append(part)
            total += len(part) + 2
        return "\n\n".join(parts)

    def _format_narrowed_examples(self, chosen: list[int], candidate_labels: list[str]) -> str:
        """Format examples organized by candidate label."""
        parts = []
        total = 0
        for label in candidate_labels:
            label_exs = [i for i in chosen if self.examples[i].get("target", "") == label]
            if not label_exs:
                continue
            section_header = f"--- Candidate: {label} ---"
            if total + len(section_header) > MAX_CHARS:
                break
            parts.append(section_header)
            total += len(section_header) + 2
            for i in label_exs:
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

        # Decide: narrowing or fallback
        use_narrowing = self._should_narrow()

        if use_narrowing:
            chosen, candidate_labels = self._narrowing_select(input)
            if not chosen:
                prompt = PASS1_TEMPLATE.format(examples_section="", input=input)
                response = self.call_llm(prompt)
                answer = extract_json_field(response, "final_answer")
                return answer, {"full_response": response, "pass": 1, "mode": "empty"}

            examples_section = self._format_narrowed_examples(chosen, candidate_labels)
            prompt1 = PASS1_TEMPLATE.format(examples_section=examples_section, input=input)
        else:
            # Fallback: standard retrieval (for unique-target tasks like USPTO)
            chosen = self._fallback_select(input)
            if not chosen:
                prompt = PASS1_FALLBACK.format(examples_section="", input=input)
                response = self.call_llm(prompt)
                answer = extract_json_field(response, "final_answer")
                return answer, {"full_response": response, "pass": 1, "mode": "empty"}

            parts = []
            total = 0
            for i in chosen:
                ex = self.examples[i]
                q = ex.get("raw_question", ex["input"])
                part = f"Q: {q}\nA: {ex['target']}"
                if total + len(part) > MAX_CHARS:
                    break
                parts.append(part)
                total += len(part) + 2
            examples_section = "\n\n".join(parts)
            prompt1 = PASS1_FALLBACK.format(examples_section=examples_section, input=input)

        response1 = self.call_llm(prompt1)
        prediction = extract_json_field(response1, "final_answer")

        if len(self.examples) < MIN_EXAMPLES_FOR_VERIFY or not prediction:
            return prediction, {"full_response": response1, "pass": 1, "mode": "narrowing" if use_narrowing else "fallback"}

        # Find alternative label (confusion-guided)
        alt_label = self._find_alternative_label(input, prediction)
        if alt_label is None:
            return prediction, {"full_response": response1, "pass": 1, "mode": "narrowing" if use_narrowing else "fallback"}

        # Pass 2: verification
        examples_a = self._get_label_examples(input, prediction, VERIFY_EXAMPLES_PER_LABEL)
        examples_b = self._get_label_examples(input, alt_label, VERIFY_EXAMPLES_PER_LABEL)

        if not examples_a or not examples_b:
            return prediction, {"full_response": response1, "pass": 1, "mode": "narrowing" if use_narrowing else "fallback"}

        prompt2 = VERIFY_TEMPLATE.format(
            prediction=prediction,
            label_a=prediction,
            label_b=alt_label,
            examples_a=self._format_examples_list(examples_a),
            examples_b=self._format_examples_list(examples_b),
            input=input,
        )
        response2 = self.call_llm(prompt2)
        final_answer = extract_json_field(response2, "final_answer")

        return final_answer or prediction, {
            "full_response": response2,
            "pass": 2,
            "pass1_prediction": prediction,
            "alternative": alt_label,
            "mode": "narrowing" if use_narrowing else "fallback",
            "confusion_guided": alt_label in self._confusion.get(prediction, {}),
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for r in batch_results:
            ex = {"input": str(r["input"]), "target": str(r["ground_truth"])}
            if "raw_question" in r:
                ex["raw_question"] = str(r["raw_question"])
            self.examples.append(ex)
            pred = r.get("prediction", "")
            true = str(r["ground_truth"])
            if pred and pred != true:
                self._confusion[pred][true] += 1
        self._index_dirty = True

    def get_state(self) -> str:
        return json.dumps({
            "examples": self.examples,
            "confusion": {k: dict(v) for k, v in self._confusion.items()},
        }, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = data.get("examples", [])
        self._confusion = defaultdict(Counter)
        for k, v in data.get("confusion", {}).items():
            self._confusion[k] = Counter(v)
        self._index_dirty = True
