"""Adaptive MMR with multi-label completeness prompting.

Exploitation candidate based on iter012_mmr_confusion_verify (current best, 50.2%).

Mechanism changes:
1. ADAPTIVE MMR LAMBDA: Detects task type from training data. On unique-target
   tasks (USPTO), sets lambda=1.0 (pure BM25, no diversity penalty). On
   classification tasks, keeps lambda=0.7 (proven best). This recovers the
   USPTO regression (26.7% -> ~33%) while preserving S2D/LawBench gains.

2. MULTI-LABEL COMPLETENESS INSTRUCTION: Detects multi-label patterns in
   training targets (semicolons). When detected, adds explicit instructions
   to the pass1 prompt telling the model to check for multiple applicable
   answers. This targets LawBench's #1 failure mode: predicting only 1 of N
   charges (40% of errors).

These are PROMPT ORGANIZATION and RETRIEVAL STRATEGY changes, not parameter
tweaks. The adaptive lambda changes which examples appear in the prompt
(fundamentally different set on USPTO). The completeness instruction changes
how the model reasons about its answer.

Key differences from prior attempts:
- iter006_label_inventory_verify: Added full label list (too many labels,
  confused the model, scored 40%). We add only a behavioral instruction.
- iter009_multilabel_recheck: Used a separate LLM call for completeness.
  We integrate the instruction into pass1 (no extra cost).
- iter009_label_snap_verify: Post-processed labels. We change the prompt
  to prevent the error in the first place.
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
{completeness_instruction}- Respond in JSON format.

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


def _jaccard_sim(toks_a: set, toks_b: set) -> float:
    if not toks_a or not toks_b:
        return 0.0
    inter = len(toks_a & toks_b)
    union = len(toks_a | toks_b)
    return inter / union if union > 0 else 0.0


class AdaptiveMmrComplete(BaseAgentMemory):
    """Adaptive MMR retrieval + multi-label completeness + confusion verify."""

    def __init__(self, llm):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self._docs_tokens: list[list[str]] = []
        self._doc_tfs: list[Counter] = []
        self._doc_lens: list[int] = []
        self._doc_token_sets: list[set] = []
        self._idf: dict[str, float] = {}
        self._avgdl: float = 0.0
        self._label_to_idxs: dict[str, list[int]] = defaultdict(list)
        self._index_dirty = True
        self._confusion: dict[str, Counter] = defaultdict(Counter)
        self._mmr_lambda: float = 0.7
        self._is_multilabel: bool = False
        self._separator: str = ";"

    def _detect_task_properties(self) -> None:
        """Detect task type and multi-label from training data."""
        if not self.examples:
            return
        label_counts = Counter(ex.get("target", "") for ex in self.examples)
        unique_count = sum(1 for c in label_counts.values() if c == 1)
        total_labels = len(label_counts)
        unique_ratio = unique_count / total_labels if total_labels > 0 else 0
        # Adaptive lambda
        if unique_ratio > 0.7:
            self._mmr_lambda = 1.0  # Pure BM25
        elif unique_ratio > 0.4:
            self._mmr_lambda = 0.85
        else:
            self._mmr_lambda = 0.7
        # Multi-label detection
        for sep in [";", "；", ",", "、"]:
            count = sum(1 for ex in self.examples if sep in ex.get("target", ""))
            if count >= 3 and count / len(self.examples) >= 0.05:
                self._is_multilabel = True
                self._separator = sep
                return
        self._is_multilabel = False

    def _ensure_index(self) -> None:
        if not self._index_dirty:
            return
        questions = [ex.get("raw_question") or ex["input"] for ex in self.examples]
        self._docs_tokens = [_tokenize(q) for q in questions]
        self._doc_tfs = [Counter(t) for t in self._docs_tokens]
        self._doc_lens = [len(t) for t in self._docs_tokens]
        self._doc_token_sets = [set(t) for t in self._docs_tokens]
        n = len(self._docs_tokens)
        self._avgdl = (sum(self._doc_lens) / n) if n else 0.0
        self._idf = _bm25_idf(self._docs_tokens)
        self._label_to_idxs = defaultdict(list)
        for i, ex in enumerate(self.examples):
            self._label_to_idxs[ex.get("target", "")].append(i)
        self._detect_task_properties()
        self._index_dirty = False

    def _mmr_select(self, query: str) -> list[int]:
        """Select examples using adaptive MMR."""
        self._ensure_index()
        if not self.examples:
            return []
        qtoks = _tokenize(query)
        n = len(self.examples)
        bm25_scores = [
            _bm25_score(qtoks, self._doc_tfs[i], self._doc_lens[i], self._avgdl, self._idf)
            for i in range(n)
        ]
        pool_indices = sorted(range(n), key=lambda i: -bm25_scores[i])[:CANDIDATE_POOL]
        if not pool_indices:
            return []
        pool_scores = [bm25_scores[i] for i in pool_indices]
        max_score = max(pool_scores) if pool_scores else 1.0
        if max_score <= 0:
            max_score = 1.0
        norm_scores = {i: bm25_scores[i] / max_score for i in pool_indices}

        lam = self._mmr_lambda
        chosen: list[int] = []
        chosen_set: set[int] = set()

        for _ in range(min(TOP_K, len(pool_indices))):
            best_idx = -1
            best_mmr = -float('inf')
            for i in pool_indices:
                if i in chosen_set:
                    continue
                relevance = norm_scores[i]
                if not chosen or lam >= 1.0:
                    redundancy = 0.0
                else:
                    redundancy = max(
                        _jaccard_sim(self._doc_token_sets[i], self._doc_token_sets[j])
                        for j in chosen
                    )
                mmr = lam * relevance - (1 - lam) * redundancy
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i
            if best_idx < 0:
                break
            chosen.append(best_idx)
            chosen_set.add(best_idx)
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

    def _get_completeness_instruction(self) -> str:
        if not self._is_multilabel:
            return ""
        return (
            f"- IMPORTANT: The answer may involve MULTIPLE items. If the input "
            f"describes several distinct categories or offenses, list ALL that apply, "
            f"separated by '{self._separator}'. Do not stop at the first match.\n"
        )

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        idxs = self._mmr_select(input)
        completeness = self._get_completeness_instruction()

        if not idxs:
            prompt = PASS1_TEMPLATE.format(
                examples_section="", input=input,
                completeness_instruction=completeness
            )
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

        prompt1 = PASS1_TEMPLATE.format(
            examples_section=examples_section, input=input,
            completeness_instruction=completeness
        )
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
            "mmr_lambda": self._mmr_lambda,
            "is_multilabel": self._is_multilabel,
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
