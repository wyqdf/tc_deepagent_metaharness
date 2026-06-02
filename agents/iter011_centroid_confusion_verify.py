"""Centroid-prior retrieval with confusion-guided verification.

Mechanism change vs iter005_confusion_verify (base, 48.4% frontier):
Replaces flat BM25 retrieval scoring with CENTROID-BOOSTED scoring.
For each label seen in training, builds a centroid term-frequency vector
(sum of TFs across all examples sharing that label). At retrieval time,
each document's score is boosted by how well the query matches the
centroid of that document's label class.

On classification tasks where labels repeat, this improves retrieval
precision. On unique-target tasks, centroid = single doc, so ranking
degrades gracefully to plain BM25.

The verification pass remains confusion-guided (same as iter005).
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
CENTROID_ALPHA = 0.6


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


class CentroidConfusionVerify(BaseAgentMemory):
    """Centroid-prior retrieval + confusion-guided two-pass verification."""

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
        # Centroid structures
        self._centroid_tfs: dict[str, Counter] = {}
        self._centroid_lens: dict[str, int] = {}
        self._centroid_avgdl: float = 0.0

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
        # Build centroids
        self._centroid_tfs = {}
        self._centroid_lens = {}
        for label, idxs in self._label_to_idxs.items():
            combined = Counter()
            for i in idxs:
                combined.update(self._doc_tfs[i])
            self._centroid_tfs[label] = combined
            self._centroid_lens[label] = sum(combined.values())
        if self._centroid_lens:
            self._centroid_avgdl = sum(self._centroid_lens.values()) / len(self._centroid_lens)
        else:
            self._centroid_avgdl = 0.0
        self._index_dirty = False

    def _get_centroid_boosted_scores(self, query: str) -> list[float]:
        """Get centroid-boosted BM25 scores for all examples."""
        qtoks = _tokenize(query)
        n = len(self.examples)
        # Instance-level scores
        doc_scores = [
            _bm25_score(qtoks, self._doc_tfs[i], self._doc_lens[i], self._avgdl, self._idf)
            for i in range(n)
        ]
        max_doc = max(doc_scores) if doc_scores else 1.0
        if max_doc <= 0:
            max_doc = 1.0
        # Centroid scores per label
        centroid_scores: dict[str, float] = {}
        for label, ctf in self._centroid_tfs.items():
            centroid_scores[label] = _bm25_score(
                qtoks, ctf, self._centroid_lens[label], self._centroid_avgdl, self._idf
            )
        max_cent = max(centroid_scores.values()) if centroid_scores else 1.0
        if max_cent <= 0:
            max_cent = 1.0
        # Combined scores
        combined = []
        for i in range(n):
            label = self.examples[i].get("target", "")
            norm_doc = doc_scores[i] / max_doc
            norm_cent = centroid_scores.get(label, 0) / max_cent
            combined.append(norm_doc + CENTROID_ALPHA * norm_cent)
        return combined

    def _select(self, query: str) -> list[int]:
        self._ensure_index()
        if not self.examples:
            return []
        scores = self._get_centroid_boosted_scores(query)
        n = len(self.examples)
        ranked = sorted(range(n), key=lambda i: -scores[i])
        pool = ranked[:CANDIDATE_POOL]
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
        # BM25 fallback
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
