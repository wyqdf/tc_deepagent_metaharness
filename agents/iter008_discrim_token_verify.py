"""Discriminative-token BM25 blend with two-pass confusion verification.

Mechanism change vs iter005_confusion_verify (base): Adds a SECOND BM25 index
built from label-discriminative tokens only. For each token in the corpus,
computes max_L P(L|t) * IDF(t) — tokens concentrated in one label get high
scores; generic tokens shared across many labels get low scores. Each document
is filtered to keep only its top-N most discriminative tokens, and a second
BM25 index is built over these filtered docs.

At retrieval time, the final score blends raw BM25 with filtered BM25:
    score = (1-beta)*raw_bm25_norm + beta*filtered_bm25_norm

This helps on datasets like LawBench where many cases share generic tokens
(被告人, 人民检察院) but differ in label-specific tokens (盗窃 vs 抢劫).
On USPTO (unique labels), discrimination scores are uniform so the filter
degrades gracefully to raw BM25.

The two-pass confusion-guided verification from iter005 is preserved on top.

Axes changed: RETRIEVAL (discriminative token filtering) — genuinely new
retrieval mechanism, not a parameter change.
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
ANCHORS = 3
PER_LABEL_CAP = 2
VERIFY_EXAMPLES_PER_LABEL = 3
MIN_EXAMPLES_FOR_VERIFY = 6
CONFUSION_MIN = 1
DISCRIM_BETA = 0.4
DISCRIM_KEEP = 60
DISCRIM_MIN_DOCS = 10


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


def _compute_discrim(docs_tokens: list[list[str]], labels: list[str]) -> dict[str, float]:
    """Per-token label discrimination: IDF * max_L P(L|t)."""
    label_tok_count: dict[str, Counter] = defaultdict(Counter)
    tok_total: Counter = Counter()
    for tokens, label in zip(docs_tokens, labels):
        for t in set(tokens):
            label_tok_count[label][t] += 1
            tok_total[t] += 1
    n = len(docs_tokens)
    discrim: dict[str, float] = {}
    for t, total in tok_total.items():
        if total < 2:
            discrim[t] = 0.0
            continue
        max_share = max(lc.get(t, 0) / total for lc in label_tok_count.values())
        idf_t = math.log(1 + (n - total + 0.5) / (total + 0.5))
        discrim[t] = idf_t * max_share
    return discrim


def _filter_doc_tokens(tokens: list[str], discrim: dict[str, float], keep: int) -> list[str]:
    unique_scored = sorted(((discrim.get(t, 0.0), t) for t in set(tokens)), reverse=True)
    keep_set = {t for _, t in unique_scored[:keep]}
    return [t for t in tokens if t in keep_set]


class DiscrimTokenVerify(BaseAgentMemory):
    """Discriminative-token BM25 blend + two-pass confusion verification."""

    def __init__(self, llm):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        # Raw index
        self._docs_tokens: list[list[str]] = []
        self._doc_tfs: list[Counter] = []
        self._doc_lens: list[int] = []
        self._idf: dict[str, float] = {}
        self._avgdl: float = 0.0
        # Filtered index
        self._filt_tfs: list[Counter] = []
        self._filt_lens: list[int] = []
        self._filt_idf: dict[str, float] = {}
        self._filt_avgdl: float = 0.0
        self._discrim: dict[str, float] = {}
        self._use_discrim: bool = False
        # Label index
        self._label_to_idxs: dict[str, list[int]] = defaultdict(list)
        self._index_dirty = True
        # Confusion tracking
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

        # Build discriminative filtered index
        labels = [ex.get("target", "") for ex in self.examples]
        unique_labels = set(labels)
        self._use_discrim = n >= DISCRIM_MIN_DOCS and len(unique_labels) < n
        if self._use_discrim:
            self._discrim = _compute_discrim(self._docs_tokens, labels)
            filt_tokens = [_filter_doc_tokens(t, self._discrim, DISCRIM_KEEP) for t in self._docs_tokens]
            self._filt_tfs = [Counter(t) for t in filt_tokens]
            self._filt_lens = [len(t) for t in filt_tokens]
            self._filt_avgdl = (sum(self._filt_lens) / n) if n else 0.0
            self._filt_idf = _bm25_idf(filt_tokens)
        self._index_dirty = False

    def _blended_scores(self, query: str) -> list[float]:
        """Compute blended raw + discriminative BM25 scores."""
        self._ensure_index()
        qtoks = _tokenize(query)
        n = len(self.examples)
        raw_scores = [
            _bm25_score(qtoks, self._doc_tfs[i], self._doc_lens[i], self._avgdl, self._idf)
            for i in range(n)
        ]
        if not self._use_discrim:
            return raw_scores
        # Filter query tokens too
        filt_qtoks = _filter_doc_tokens(qtoks, self._discrim, DISCRIM_KEEP)
        if not filt_qtoks:
            filt_qtoks = qtoks
        filt_scores = [
            _bm25_score(filt_qtoks, self._filt_tfs[i], self._filt_lens[i], self._filt_avgdl, self._filt_idf)
            for i in range(n)
        ]
        # Normalize both
        raw_max = max(raw_scores) if raw_scores else 1.0
        filt_max = max(filt_scores) if filt_scores else 1.0
        raw_max = raw_max if raw_max > 0 else 1.0
        filt_max = filt_max if filt_max > 0 else 1.0
        return [
            (1 - DISCRIM_BETA) * (r / raw_max) + DISCRIM_BETA * (f / filt_max)
            for r, f in zip(raw_scores, filt_scores)
        ]

    def _select(self, query: str) -> list[int]:
        self._ensure_index()
        if not self.examples:
            return []
        scores = self._blended_scores(query)
        n = len(self.examples)
        pool = sorted(range(n), key=lambda i: -scores[i])[:CANDIDATE_POOL]
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

    def _find_alternative_label(self, query: str, predicted: str) -> str | None:
        if predicted in self._confusion:
            candidates = self._confusion[predicted].most_common()
            for label, count in candidates:
                if count >= CONFUSION_MIN and label in self._label_to_idxs:
                    return label
        self._ensure_index()
        scores = self._blended_scores(query)
        label_best: dict[str, float] = {}
        for i in range(len(self.examples)):
            label = self.examples[i].get("target", "")
            if label == predicted:
                continue
            if label not in label_best or scores[i] > label_best[label]:
                label_best[label] = scores[i]
        if not label_best:
            return None
        return max(label_best, key=label_best.get)

    def _get_label_examples(self, query: str, label: str, k: int) -> list[int]:
        idxs = self._label_to_idxs.get(label, [])
        if not idxs:
            return []
        scores = self._blended_scores(query)
        scored = [(scores[i], i) for i in idxs]
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
        idxs = self._select(input)
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
