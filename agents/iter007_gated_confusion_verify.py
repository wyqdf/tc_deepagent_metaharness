"""Gated confusion verification with multi-alternative comparison.

Mechanism change vs iter005_confusion_verify (base): Two key changes:

1. CONFIDENCE GATING: Only triggers the expensive verification pass when the
   predicted label has known confusion history (>= 2 errors in training).
   Labels that have never been confused skip verification entirely. This
   prevents the verification pass from corrupting correct predictions —
   a failure mode observed in iter005/iter006 where verification changed
   correct answers to wrong ones (especially on Symptom2Disease).

2. MULTI-ALTERNATIVE VERIFICATION: When confusion data provides 2+ alternative
   labels, the verification prompt compares against ALL top alternatives (up to 3)
   instead of just 1. This increases the chance the correct answer appears in
   the verification set.

These are LEARNING STRATEGY + VERIFICATION mechanism changes:
- The gating uses accumulated confusion statistics to decide WHETHER to verify
- The multi-alt uses confusion ranking to decide WHAT to verify against
- Both change the verification behavior, not just parameters
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

VERIFY_TEMPLATE_2 = """You previously predicted the answer below. Now verify it by comparing against focused examples for two candidate answers.

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

VERIFY_TEMPLATE_3 = """You previously predicted the answer below. Now verify by comparing against focused examples for the most likely candidates.

**Your initial prediction:** {prediction}

**Examples for "{label_a}":**
{examples_a}

**Examples for "{label_b}":**
{examples_b}

**Examples for "{label_c}":**
{examples_c}

**Problem:**
{input}

**Instructions:**
- Compare the problem carefully against all three sets of examples.
- The correct answer is most likely one of: "{label_a}", "{label_b}", or "{label_c}" (or something else if none fits).
- If your initial prediction was wrong, correct it now.
- Respond in JSON format.

{{"reasoning": "[compare against all candidate labels]", "final_answer": "[your final answer]"}}"""

MAX_CHARS = 28000
TOP_K = 14
CANDIDATE_POOL = 64
NGRAM_MIN = 2
NGRAM_MAX = 3
ANCHORS = 3
PER_LABEL_CAP = 2
VERIFY_EXAMPLES_PER_LABEL = 3
MIN_EXAMPLES_FOR_VERIFY = 6
CONFUSION_GATE_MIN = 2  # minimum confusion count to trigger verification


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


class GatedConfusionVerify(BaseAgentMemory):
    """Confidence-gated two-pass verify with multi-alternative comparison."""

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
        # Confusion tracking: predicted_label -> {true_label: count}
        self._confusion: dict[str, Counter] = defaultdict(Counter)
        # Track total predictions per label for gating
        self._pred_counts: Counter = Counter()

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

    def _should_verify(self, prediction: str) -> tuple[bool, list[str]]:
        """Gating: only verify if prediction has sufficient confusion history."""
        if prediction not in self._confusion:
            return False, []
        total_confused = sum(self._confusion[prediction].values())
        if total_confused < CONFUSION_GATE_MIN:
            return False, []
        # Get top alternatives (up to 2)
        alternatives = [
            label for label, _ in self._confusion[prediction].most_common(3)
            if label in self._label_to_idxs and self._label_to_idxs[label]
        ]
        if not alternatives:
            return False, []
        return True, alternatives[:2]

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
            if total + len(part) > MAX_CHARS // 3:
                break
            parts.append(part)
            total += len(part) + 2
        return "\n\n".join(parts)

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        # Pass 1: standard retrieval + prediction
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

        # Confidence gating: only verify if prediction has confusion history
        should_verify, alternatives = self._should_verify(prediction)
        if not should_verify:
            return prediction, {"full_response": response1, "pass": 1, "gated": True}

        # Pass 2: verification with alternatives
        examples_pred = self._get_label_examples(input, prediction, VERIFY_EXAMPLES_PER_LABEL)
        if not examples_pred:
            return prediction, {"full_response": response1, "pass": 1}

        examples_alt1 = self._get_label_examples(input, alternatives[0], VERIFY_EXAMPLES_PER_LABEL)
        if not examples_alt1:
            return prediction, {"full_response": response1, "pass": 1}

        if len(alternatives) >= 2:
            examples_alt2 = self._get_label_examples(input, alternatives[1], VERIFY_EXAMPLES_PER_LABEL)
            if examples_alt2:
                # 3-way comparison
                prompt2 = VERIFY_TEMPLATE_3.format(
                    prediction=prediction,
                    label_a=prediction,
                    label_b=alternatives[0],
                    label_c=alternatives[1],
                    examples_a=self._format_examples_list(examples_pred),
                    examples_b=self._format_examples_list(examples_alt1),
                    examples_c=self._format_examples_list(examples_alt2),
                    input=input,
                )
            else:
                # 2-way comparison
                prompt2 = VERIFY_TEMPLATE_2.format(
                    prediction=prediction,
                    label_a=prediction,
                    label_b=alternatives[0],
                    examples_a=self._format_examples_list(examples_pred),
                    examples_b=self._format_examples_list(examples_alt1),
                    input=input,
                )
        else:
            # 2-way comparison
            prompt2 = VERIFY_TEMPLATE_2.format(
                prediction=prediction,
                label_a=prediction,
                label_b=alternatives[0],
                examples_a=self._format_examples_list(examples_pred),
                examples_b=self._format_examples_list(examples_alt1),
                input=input,
            )

        response2 = self.call_llm(prompt2)
        final_answer = extract_json_field(response2, "final_answer")

        return final_answer or prediction, {
            "full_response": response2,
            "pass": 2,
            "pass1_prediction": prediction,
            "alternatives": alternatives,
            "gated": False,
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for r in batch_results:
            ex = {"input": str(r["input"]), "target": str(r["ground_truth"])}
            if "raw_question" in r:
                ex["raw_question"] = str(r["raw_question"])
            self.examples.append(ex)
            # Track confusion from predictions
            pred = r.get("prediction", "")
            true = str(r["ground_truth"])
            if pred:
                self._pred_counts[pred] += 1
                if pred != true:
                    self._confusion[pred][true] += 1
        self._index_dirty = True

    def get_state(self) -> str:
        return json.dumps({
            "examples": self.examples,
            "confusion": {k: dict(v) for k, v in self._confusion.items()},
            "pred_counts": dict(self._pred_counts),
        }, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = data.get("examples", [])
        self._confusion = defaultdict(Counter)
        for k, v in data.get("confusion", {}).items():
            self._confusion[k] = Counter(v)
        self._pred_counts = Counter(data.get("pred_counts", {}))
        self._index_dirty = True
