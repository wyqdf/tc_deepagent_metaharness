"""Error-prior boosted retrieval with confusion-guided verification.

Exploration candidate. Fundamentally different retrieval mechanism from MMR.

Mechanism: Instead of diversity-based selection (MMR), uses ERROR-PRIOR
BOOSTED retrieval. Tracks per-label error rates during training. At predict
time, after BM25 scoring, boosts scores of examples whose labels have high
error rates. This ensures the model sees examples from labels it struggles
with, even if they aren't the most BM25-similar.

Key insight: The model's errors are not random — they cluster around specific
confusable label pairs. By boosting representation of high-error labels in
the prompt, we give the model more evidence to discriminate these hard cases.

Differences from prior systems:
- iter002_error_weighted_bm25: Boosted individual misclassified EXAMPLES.
  This system boosts entire LABEL CATEGORIES based on aggregate error rates.
- iter012_mmr_confusion_verify: Uses content diversity (Jaccard).
  This system uses error-informed diversity (label difficulty).
- iter001_confusion_aware_bm25: Tracked confusion pairs for retrieval.
  This system uses per-label error rates as a prior on retrieval scores.

The error-prior acts as a soft constraint: labels the model gets wrong more
often get a score boost proportional to their error rate, ensuring at least
some examples from hard labels appear in the prompt. On tasks where all
labels have similar error rates (or no errors yet), degrades to pure BM25.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from typing import Any

from harness.agent_protocol import BaseAgentMemory, extract_json_field

PASS1_TEMPLATE = """Solve the problem below based on the examples provided.

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
TOP_K = 14
CANDIDATE_POOL = 64
NGRAM_MIN = 2
NGRAM_MAX = 3
VERIFY_EXAMPLES_PER_LABEL = 3
MIN_EXAMPLES_FOR_VERIFY = 6
CONFUSION_MIN = 1
ERROR_BOOST_ALPHA = 0.3  # How much to boost high-error labels


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


class ErrorPriorRetrieval(BaseAgentMemory):
    """Error-prior boosted retrieval + confusion-guided verification."""

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
        self._label_error_count: Counter = Counter()
        self._label_total_count: Counter = Counter()

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

    def _compute_error_prior(self) -> dict[str, float]:
        """Compute per-label error rate as a retrieval boost prior."""
        if not self._label_total_count:
            return {}
        priors = {}
        max_rate = 0.0
        for label in self._label_total_count:
            total = self._label_total_count[label]
            errors = self._label_error_count[label]
            rate = errors / total if total > 0 else 0.0
            priors[label] = rate
            if rate > max_rate:
                max_rate = rate
        # Normalize to [0, 1]
        if max_rate > 0:
            priors = {k: v / max_rate for k, v in priors.items()}
        return priors

    def _error_boosted_select(self, query: str) -> list[int]:
        """Select examples with error-prior boosted BM25 scores."""
        self._ensure_index()
        if not self.examples:
            return []
        qtoks = _tokenize(query)
        n = len(self.examples)

        # Compute error priors
        error_priors = self._compute_error_prior()

        # Score all docs with error-prior boost
        scores = []
        for i in range(n):
            bm25 = _bm25_score(
                qtoks, self._doc_tfs[i], self._doc_lens[i], self._avgdl, self._idf
            )
            label = self.examples[i].get("target", "")
            prior = error_priors.get(label, 0.0)
            # Boosted score: BM25 * (1 + alpha * error_prior)
            boosted = bm25 * (1.0 + ERROR_BOOST_ALPHA * prior)
            scores.append((boosted, bm25, i))

        scores.sort(reverse=True)

        # Select top-K with label coverage constraint
        chosen = []
        label_counts: Counter = Counter()
        per_label_cap = 3

        for _, _, i in scores[:CANDIDATE_POOL]:
            if len(chosen) >= TOP_K:
                break
            label = self.examples[i].get("target", "")
            if label_counts[label] >= per_label_cap:
                continue
            chosen.append(i)
            label_counts[label] += 1

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

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        idxs = self._error_boosted_select(input)

        if not idxs:
            prompt = PASS1_TEMPLATE.format(examples_section="", input=input)
            response = self.call_llm(prompt)
            answer = extract_json_field(response, "final_answer")
            return answer, {"full_response": response, "pass": 1}

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
        examples_section = "\n\n".join(parts)

        prompt1 = PASS1_TEMPLATE.format(examples_section=examples_section, input=input)
        response1 = self.call_llm(prompt1)
        prediction = extract_json_field(response1, "final_answer")

        if len(self.examples) < MIN_EXAMPLES_FOR_VERIFY or not prediction:
            return prediction, {"full_response": response1, "pass": 1}

        alt_label = self._find_alternative_label(input, prediction)
        if alt_label is None:
            return prediction, {"full_response": response1, "pass": 1}

        examples_a = self._get_label_examples(input, prediction, VERIFY_EXAMPLES_PER_LABEL)
        examples_b = self._get_label_examples(input, alt_label, VERIFY_EXAMPLES_PER_LABEL)

        if not examples_a or not examples_b:
            return prediction, {"full_response": response1, "pass": 1}

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
            # Track label totals and errors
            self._label_total_count[true] += 1
            if pred and pred != true:
                self._confusion[pred][true] += 1
                self._label_error_count[true] += 1
                # Also count the predicted label as involved in error
                self._label_error_count[pred] += 1
        self._index_dirty = True

    def get_state(self) -> str:
        return json.dumps({
            "examples": self.examples,
            "confusion": {k: dict(v) for k, v in self._confusion.items()},
            "label_error_count": dict(self._label_error_count),
            "label_total_count": dict(self._label_total_count),
        }, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = data.get("examples", [])
        self._confusion = defaultdict(Counter)
        for k, v in data.get("confusion", {}).items():
            self._confusion[k] = Counter(v)
        self._label_error_count = Counter(data.get("label_error_count", {}))
        self._label_total_count = Counter(data.get("label_total_count", {}))
        self._index_dirty = True
