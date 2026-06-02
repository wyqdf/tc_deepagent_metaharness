"""MMR retrieval with multi-alternative verification.

Mechanism change vs iter012_mmr_confusion_verify (base): Expands the verification
step from binary (predicted vs 1 alternative) to multi-alternative (predicted vs
top-3 confused labels). This gives the model more correction options.

Key insight: The binary verify in iter012 can only correct to ONE alternative.
On LawBench, the model often predicts a close-but-wrong crime name, and the
correct answer might not be the single most-confused alternative. By showing
3 alternatives, we increase the chance the correct label is among the options.

On S2D: if the model confuses Disease A with B, C, and D, showing all three
alternatives lets it pick the right correction.

On USPTO (unique targets): verify rarely triggers (no repeated labels to build
confusion history), so this change has minimal impact there.

Falls back to BM25-based alternatives when confusion data is sparse.
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

VERIFY_TEMPLATE = """You previously predicted the answer below. Now verify it by comparing against focused examples for multiple candidate answers.

**Your initial prediction:** {prediction}

{candidates_section}

**Problem:**
{input}

**Instructions:**
- Compare the problem carefully against all candidate answer sets above.
- The correct answer is most likely one of the candidates shown, but could be something else if none fits.
- If your initial prediction was wrong, correct it now.
- Respond in JSON format.

{{"reasoning": "[compare against all candidate labels]", "final_answer": "[your final answer]"}}"""

MAX_CHARS = 28000
TOP_K = 14
CANDIDATE_POOL = 64
NGRAM_MIN = 2
NGRAM_MAX = 3
MMR_LAMBDA = 0.7
VERIFY_EXAMPLES_PER_LABEL = 2
MIN_EXAMPLES_FOR_VERIFY = 6
CONFUSION_MIN = 1
MAX_VERIFY_ALTS = 3


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


class MultiAltVerify(BaseAgentMemory):
    """MMR-diversified retrieval + multi-alternative verification."""

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
        self._index_dirty = False

    def _mmr_select(self, query: str) -> list[int]:
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

        max_bm25 = max(bm25_scores[i] for i in pool_indices) or 1.0
        selected: list[int] = []
        selected_sets: list[set] = []
        remaining = set(pool_indices)

        while len(selected) < TOP_K and remaining:
            best_idx = -1
            best_score = -float("inf")
            for idx in remaining:
                rel = bm25_scores[idx] / max_bm25
                red = 0.0
                if selected_sets:
                    red = max(_jaccard_sim(self._doc_token_sets[idx], s) for s in selected_sets)
                score = MMR_LAMBDA * rel - (1 - MMR_LAMBDA) * red
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx < 0:
                break
            selected.append(best_idx)
            selected_sets.append(self._doc_token_sets[best_idx])
            remaining.discard(best_idx)
        return selected

    def _find_alternative_labels(self, query: str, prediction: str) -> list[str]:
        """Find top alternative labels using confusion history + BM25 fallback."""
        alts = []
        # From confusion history
        if prediction in self._confusion:
            confused = self._confusion[prediction].most_common()
            for label, count in confused:
                if count >= CONFUSION_MIN and label != prediction:
                    alts.append(label)
                    if len(alts) >= MAX_VERIFY_ALTS:
                        return alts

        # BM25 fallback for remaining slots
        if len(alts) < MAX_VERIFY_ALTS:
            self._ensure_index()
            qtoks = _tokenize(query)
            label_scores: dict[str, float] = defaultdict(float)
            for label, idxs in self._label_to_idxs.items():
                if label == prediction or label in alts:
                    continue
                for i in idxs:
                    s = _bm25_score(qtoks, self._doc_tfs[i], self._doc_lens[i], self._avgdl, self._idf)
                    label_scores[label] = max(label_scores[label], s)
            ranked = sorted(label_scores.items(), key=lambda x: -x[1])
            for label, _ in ranked:
                if label not in alts:
                    alts.append(label)
                    if len(alts) >= MAX_VERIFY_ALTS:
                        break
        return alts

    def _get_label_examples(self, query: str, label: str, k: int) -> list[int]:
        self._ensure_index()
        idxs = self._label_to_idxs.get(label, [])
        if not idxs:
            return []
        qtoks = _tokenize(query)
        scored = []
        for i in idxs:
            s = _bm25_score(qtoks, self._doc_tfs[i], self._doc_lens[i], self._avgdl, self._idf)
            scored.append((s, i))
        scored.sort(reverse=True)
        return [i for _, i in scored[:k]]

    def _format_examples_list(self, idxs: list[int], budget: int) -> str:
        parts = []
        total = 0
        for i in idxs:
            ex = self.examples[i]
            q = ex.get("raw_question", ex["input"])
            part = f"Q: {q}\nA: {ex['target']}"
            if total + len(part) > budget:
                break
            parts.append(part)
            total += len(part) + 2
        return "\n\n".join(parts)

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        # Pass 1: MMR-diversified retrieval + prediction
        idxs = self._mmr_select(input)
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

        # Find alternative labels (multi-alternative)
        alt_labels = self._find_alternative_labels(input, prediction)
        if not alt_labels:
            return prediction, {"full_response": response1, "pass": 1}

        # Pass 2: multi-alternative verification
        per_label_budget = MAX_CHARS // (len(alt_labels) + 1)
        candidates_section_parts = []

        # Examples for predicted label
        pred_examples = self._get_label_examples(input, prediction, VERIFY_EXAMPLES_PER_LABEL)
        if pred_examples:
            formatted = self._format_examples_list(pred_examples, per_label_budget)
            candidates_section_parts.append(f'**Examples for "{prediction}":**\n{formatted}')

        # Examples for each alternative
        for alt in alt_labels:
            alt_examples = self._get_label_examples(input, alt, VERIFY_EXAMPLES_PER_LABEL)
            if alt_examples:
                formatted = self._format_examples_list(alt_examples, per_label_budget)
                candidates_section_parts.append(f'**Examples for "{alt}":**\n{formatted}')

        if len(candidates_section_parts) < 2:
            return prediction, {"full_response": response1, "pass": 1}

        candidates_section = "\n\n".join(candidates_section_parts)
        prompt2 = VERIFY_TEMPLATE.format(
            prediction=prediction,
            candidates_section=candidates_section,
            input=input,
        )
        response2 = self.call_llm(prompt2)
        final_answer = extract_json_field(response2, "final_answer")

        return final_answer or prediction, {
            "full_response": response2,
            "pass": 2,
            "pass1_prediction": prediction,
            "alternatives": alt_labels,
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
