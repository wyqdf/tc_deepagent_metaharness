"""Completeness-recheck verification for multi-label tasks.

Mechanism change vs iter005_confusion_verify (base): Replaces the binary
confusion-guided verification with a COMPLETENESS RECHECK approach.

Key insight: On multi-label tasks (like LawBench where targets can be
"盗窃;故意伤害"), the model often predicts only 1 of N charges. The
confusion-verify approach compares 2 candidate labels, but this doesn't
help when the error is MISSING labels rather than WRONG labels.

This system:
1. Detects multi-label tasks from training data (targets containing ';')
2. For multi-label tasks: after initial prediction, runs a second pass
   asking "is this answer complete? Are there additional items clearly
   described in the input?" with examples showing multi-label patterns.
3. For single-label tasks: runs the standard confusion-guided verify
   (proven to work on S2D).
4. Applies label-snap post-processing to fix near-miss label names.

This is fundamentally different from confusion_verify (binary label
comparison) and from multi_strategy_vote (same question, different
retrievals). It's a RECALL-oriented second pass that targets the
specific failure mode of incomplete multi-label predictions.

Axes changed: VERIFICATION STRATEGY — completeness recheck for multi-label,
confusion verify for single-label. Adaptive per-task verification.
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

RECHECK_TEMPLATE = """You previously gave the following tentative answer to the problem.
Now check whether the answer is COMPLETE.

**Multi-label examples from training (showing format):**
{multilabel_examples}

**Problem:**
{input}

**Tentative answer:** {tentative}

**Instructions:**
- Re-read the problem carefully. The answer may contain MULTIPLE items separated by semicolons (;).
- The examples above show how multi-item answers are formatted.
- If the tentative answer is missing items that are CLEARLY supported by the input, output the corrected COMPLETE answer with all items separated by semicolons.
- If the tentative answer is already complete, output it unchanged.
- Do NOT invent items that are not clearly described in the input.
- Respond in JSON format.

{{"reasoning": "[check completeness]", "final_answer": "[your complete answer]"}}"""

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
SNAP_THRESHOLD = 0.5
MULTILABEL_RATIO_THRESHOLD = 0.1


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


def _char_bigrams(s: str) -> set[str]:
    s = s.strip()
    return {s[i:i+2] for i in range(len(s)-1)} if len(s) >= 2 else {s}


def _bigram_jaccard(a: str, b: str) -> float:
    sa, sb = _char_bigrams(a), _char_bigrams(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


class MultiLabelRecheck(BaseAgentMemory):
    """Adaptive verify: completeness recheck for multi-label, confusion verify for single."""

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
        self._known_labels: set[str] = set()
        self._has_multilabel: bool = False
        # Indices of training examples with multi-label targets
        self._multilabel_idxs: list[int] = []

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

    def _snap_single(self, pred: str) -> str:
        if pred in self._known_labels:
            return pred
        best_label = pred
        best_sim = 0.0
        for label in self._known_labels:
            sim = _bigram_jaccard(pred, label)
            if sim > best_sim:
                best_sim = sim
                best_label = label
        return best_label if best_sim >= SNAP_THRESHOLD else pred

    def _snap_label(self, prediction: str) -> str:
        if not self._known_labels:
            return prediction
        if self._has_multilabel and ';' in prediction:
            components = [c.strip() for c in prediction.split(';') if c.strip()]
            snapped = [self._snap_single(c) for c in components]
            return ';'.join(snapped)
        return self._snap_single(prediction)

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

    def _find_alternative_label(self, query: str, predicted: str) -> str | None:
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

    def _get_multilabel_examples_text(self, query: str, k: int = 4) -> str:
        """Get formatted multi-label training examples for the recheck prompt."""
        if not self._multilabel_idxs:
            return ""
        # Pick the most relevant multi-label examples
        self._ensure_index()
        qtoks = _tokenize(query)
        scored = [
            (_bm25_score(qtoks, self._doc_tfs[i], self._doc_lens[i], self._avgdl, self._idf), i)
            for i in self._multilabel_idxs
        ]
        scored.sort(reverse=True)
        parts = []
        total = 0
        for _, i in scored[:k]:
            ex = self.examples[i]
            q = ex.get("raw_question", ex["input"])
            # Truncate long questions
            if len(q) > 200:
                q = q[:200] + "..."
            part = f"Q: {q}\nA: {ex['target']}"
            if total + len(part) > 3000:
                break
            parts.append(part)
            total += len(part) + 2
        return "\n\n".join(parts)

    def _verify_confusion(self, input: str, prediction: str, response1: str) -> tuple[str, dict]:
        """Standard confusion-guided verification (for single-label tasks)."""
        alt_label = self._find_alternative_label(input, prediction)
        if alt_label is None:
            prediction = self._snap_label(prediction)
            return prediction, {"full_response": response1, "pass": 1}

        examples_a = self._get_label_examples(input, prediction, VERIFY_EXAMPLES_PER_LABEL)
        examples_b = self._get_label_examples(input, alt_label, VERIFY_EXAMPLES_PER_LABEL)

        if not examples_a or not examples_b:
            prediction = self._snap_label(prediction)
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
        final_answer = self._snap_label(final_answer or prediction)
        return final_answer, {
            "full_response": response2,
            "pass": 2,
            "verify_type": "confusion",
            "pass1_prediction": prediction,
            "alternative": alt_label,
        }

    def _verify_completeness(self, input: str, prediction: str, response1: str) -> tuple[str, dict]:
        """Completeness recheck verification (for multi-label tasks)."""
        ml_examples = self._get_multilabel_examples_text(input)
        if not ml_examples:
            prediction = self._snap_label(prediction)
            return prediction, {"full_response": response1, "pass": 1}

        prompt2 = RECHECK_TEMPLATE.format(
            multilabel_examples=ml_examples,
            input=input,
            tentative=prediction,
        )
        response2 = self.call_llm(prompt2)
        final_answer = extract_json_field(response2, "final_answer")
        final_answer = self._snap_label(final_answer or prediction)
        return final_answer, {
            "full_response": response2,
            "pass": 2,
            "verify_type": "completeness",
            "pass1_prediction": prediction,
        }

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        idxs = self._select(input)
        if not idxs:
            prompt = PASS1_TEMPLATE.format(examples_section="", input=input)
            response = self.call_llm(prompt)
            answer = extract_json_field(response, "final_answer")
            answer = self._snap_label(answer) if answer else answer
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
            prediction = self._snap_label(prediction) if prediction else prediction
            return prediction, {"full_response": response1, "pass": 1}

        # Choose verification strategy based on task type
        if self._has_multilabel:
            return self._verify_completeness(input, prediction, response1)
        else:
            return self._verify_confusion(input, prediction, response1)

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for r in batch_results:
            target = str(r["ground_truth"])
            ex = {"input": str(r["input"]), "target": target}
            if "raw_question" in r:
                ex["raw_question"] = str(r["raw_question"])
            idx = len(self.examples)
            self.examples.append(ex)
            # Track known labels and multi-label status
            if ';' in target:
                self._has_multilabel = True
                self._multilabel_idxs.append(idx)
                for comp in target.split(';'):
                    comp = comp.strip()
                    if comp:
                        self._known_labels.add(comp)
            else:
                self._known_labels.add(target)
            # Confusion tracking
            pred = r.get("prediction", "")
            if pred and pred != target:
                self._confusion[pred][target] += 1
        self._index_dirty = True

    def get_state(self) -> str:
        return json.dumps({
            "examples": self.examples,
            "confusion": {k: dict(v) for k, v in self._confusion.items()},
            "known_labels": sorted(self._known_labels),
            "has_multilabel": self._has_multilabel,
            "multilabel_idxs": self._multilabel_idxs,
        }, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = data.get("examples", [])
        self._confusion = defaultdict(Counter)
        for k, v in data.get("confusion", {}).items():
            self._confusion[k] = Counter(v)
        self._known_labels = set(data.get("known_labels", []))
        self._has_multilabel = data.get("has_multilabel", False)
        self._multilabel_idxs = data.get("multilabel_idxs", [])
        self._index_dirty = True
