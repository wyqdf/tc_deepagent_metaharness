"""Confusion-verify with distilled per-class playbooks.

Mechanism change vs iter005_confusion_verify (base, 48.4%): Adds LLM-distilled
per-class playbooks as a preamble to the pass1 prompt. After training, for each
label with 2+ examples, the system generates a short playbook describing the
distinguishing features of that label. At predict time, playbooks for candidate
labels (those appearing in retrieved examples) are injected before the examples.

This improves pass1 accuracy by giving the model explicit label-boundary knowledge,
reducing the number of cases that need pass2 verification. The verify pass is
still available for cases where pass1 is uncertain.

Key difference from iter004_error_notes: error notes describe confusion PAIRS
(how to distinguish A from B). Playbooks describe individual LABELS (what makes
this label distinctive from everything else). They are complementary signals.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from typing import Any

from harness.agent_protocol import BaseAgentMemory, extract_json_field

PASS1_TEMPLATE = """Solve the problem below based on the playbooks and examples provided.

{lessons_section}{examples_section}

**Problem:**
{input}

**Instructions:**
- The playbooks (if any) describe distinguishing features for candidate answers.
- The examples are the most relevant prior cases.
- Use both to commit to one answer.
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

DISTILL_PROMPT = """You are reading several solved cases that all share the same answer:

ANSWER: {label}

CASES:
{cases}

Write a short, concrete playbook (<=60 words) listing the most reliable features a new case must have to receive this answer. Include key phrasings, keywords, or structural patterns. Do not restate the answer. Respond in JSON.

{{"playbook": "[playbook text]"}}"""

MAX_CHARS = 26000
TOP_K = 12
CANDIDATE_POOL = 64
NGRAM_MIN = 2
NGRAM_MAX = 3
ANCHORS = 3
PER_LABEL_CAP = 2
VERIFY_EXAMPLES_PER_LABEL = 3
MIN_EXAMPLES_FOR_VERIFY = 6
CONFUSION_MIN = 1
MIN_EXAMPLES_PER_LABEL = 2
MAX_CASES_PER_LESSON = 4
LESSONS_BUDGET = 4000


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


class LessonsConfusionVerify(BaseAgentMemory):
    """Confusion-verify with distilled per-class playbooks."""

    def __init__(self, llm):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self.lessons: dict[str, str] = {}
        self._lessons_dirty = True
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

    def _distill_lessons(self) -> None:
        if not self._lessons_dirty:
            return
        self._lessons_dirty = False
        self._ensure_index()
        for label, idxs in self._label_to_idxs.items():
            if label in self.lessons or len(idxs) < MIN_EXAMPLES_PER_LABEL:
                continue
            cases_parts = []
            for i in idxs[:MAX_CASES_PER_LESSON]:
                ex = self.examples[i]
                q = ex.get("raw_question", ex["input"])
                cases_parts.append(f"- {q[:400]}")
            cases_text = "\n".join(cases_parts)
            prompt = DISTILL_PROMPT.format(label=label, cases=cases_text)
            response = self.call_llm(prompt)
            playbook = extract_json_field(response, "playbook")
            if playbook:
                self.lessons[label] = playbook

    def _format_lessons(self, candidate_labels: list[str]) -> str:
        if not self.lessons:
            return ""
        seen: set[str] = set()
        rendered = []
        total = 0
        for lab in candidate_labels:
            if lab in seen or lab not in self.lessons:
                continue
            seen.add(lab)
            entry = f"- {lab}: {self.lessons[lab]}"
            if total + len(entry) > LESSONS_BUDGET:
                break
            rendered.append(entry)
            total += len(entry) + 1
        if not rendered:
            return ""
        return "**Playbooks (distinguishing features per answer):**\n" + "\n".join(rendered) + "\n\n"

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

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        self._distill_lessons()
        idxs = self._select(input)
        if not idxs:
            prompt = PASS1_TEMPLATE.format(lessons_section="", examples_section="", input=input)
            response = self.call_llm(prompt)
            answer = extract_json_field(response, "final_answer")
            return answer, {"full_response": response, "pass": 1}

        # Format examples
        parts = []
        total = 0
        for i in idxs:
            ex = self.examples[i]
            q = ex.get("raw_question", ex["input"])
            part = f"Q: {q}\nA: {ex['target']}"
            if total + len(part) > MAX_CHARS - LESSONS_BUDGET:
                break
            parts.append(part)
            total += len(part) + 2
        examples_section = "\n\n".join(parts)

        # Format lessons for candidate labels
        candidate_labels = [self.examples[i].get("target", "") for i in idxs]
        lessons_section = self._format_lessons(candidate_labels)

        prompt1 = PASS1_TEMPLATE.format(
            lessons_section=lessons_section, examples_section=examples_section, input=input
        )
        response1 = self.call_llm(prompt1)
        prediction = extract_json_field(response1, "final_answer")

        if len(self.examples) < MIN_EXAMPLES_FOR_VERIFY or not prediction:
            return prediction, {"full_response": response1, "pass": 1}

        alt_label = self._find_alternative_label(input, prediction)
        if alt_label is None:
            return prediction, {"full_response": response1, "pass": 1}

        # Pass 2: verification
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
            "lessons_used": bool(lessons_section),
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
        self._lessons_dirty = True

    def get_state(self) -> str:
        return json.dumps({
            "examples": self.examples,
            "lessons": self.lessons,
            "confusion": {k: dict(v) for k, v in self._confusion.items()},
        }, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = data.get("examples", [])
        self.lessons = data.get("lessons", {})
        self._confusion = defaultdict(Counter)
        for k, v in data.get("confusion", {}).items():
            self._confusion[k] = Counter(v)
        self._index_dirty = True
        self._lessons_dirty = not bool(self.lessons)
