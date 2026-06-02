"""Inline discrimination rules with single-pass prediction.

Mechanism change vs iter005_confusion_verify (base): Fundamentally different
approach — SINGLE PASS with discrimination rules embedded in the prompt.

During learn_from_batch, tracks confusion pairs (predicted X, was actually Y).
When a confusion pair accumulates 2+ errors, generates a short LLM-written
discrimination rule explaining how to tell X from Y.

At predict time:
1. BM25 retrieval + label-coverage selection (same as base)
2. Identify candidate labels from retrieved examples
3. Look up discrimination rules relevant to those candidate labels
4. Render rules BEFORE examples in the prompt as priming
5. Single LLM call (no verification pass)

Key differences from prior systems:
- iter004_error_notes_retrieval: stored raw error notes, used separate section,
  still used two-pass. This uses LLM-GENERATED rules, places them as priming,
  and is single-pass.
- iter005_confusion_verify: two-pass with verification. This is single-pass
  with rules integrated into the primary reasoning.
- iter006_label_inventory_verify: listed ALL labels (too noisy). This only
  shows rules for RELEVANT confusion pairs from retrieved candidates.

The single-pass design avoids the failure mode where verification corrupts
correct answers, while still providing disambiguation guidance.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from typing import Any

from harness.agent_protocol import BaseAgentMemory, extract_json_field

PROMPT_WITH_RULES = """Solve the problem below based on the examples provided.

**Disambiguation rules (from past errors — pay attention to these):**
{rules_section}

{examples_section}

**Problem:**
{input}

**Instructions:**
- The disambiguation rules above highlight common mistakes. Use them to avoid known confusions.
- The examples span the most plausible answers. Compare the problem to each example and pick the answer whose example best matches.
- Follow the patterns shown in the examples.
- Respond in JSON format.

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""

PROMPT_NO_RULES = """Solve the problem below based on the examples provided.

{examples_section}

**Problem:**
{input}

**Instructions:**
- The examples above span the most plausible answers. Compare the problem to each example and pick the answer whose example best matches.
- Follow the patterns shown in the examples.
- Respond in JSON format.

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""

RULE_GEN_PROMPT = """Two answers are often confused with each other. Write ONE short sentence (max 30 words) explaining the key difference that distinguishes them.

Answer A: {label_a}
Answer B: {label_b}

Example cases where A was correct:
{cases_a}

Example cases where B was correct:
{cases_b}

Respond in JSON: {{"rule": "[one sentence distinguishing A from B]"}}"""

MAX_CHARS = 28000
TOP_K = 14
CANDIDATE_POOL = 64
NGRAM_MIN = 2
NGRAM_MAX = 3
ANCHORS = 3
PER_LABEL_CAP = 2
CONFUSION_THRESHOLD = 2  # min errors before generating a rule
MAX_RULES_IN_PROMPT = 4  # max rules to show
MAX_CASES_FOR_RULE = 2   # examples per label for rule generation


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


class InlineRulesRetrieval(BaseAgentMemory):
    """Single-pass BM25 retrieval with inline discrimination rules."""

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
        # Confusion tracking: (pred_label, true_label) -> count
        self._confusion_counts: dict[str, Counter] = defaultdict(Counter)
        # Generated rules: (label_a, label_b) -> rule text
        self._rules: dict[tuple[str, str], str] = {}
        # Track which pairs we've already tried to generate rules for
        self._rule_generated: set[tuple[str, str]] = set()

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

    def _generate_pending_rules(self) -> None:
        """Generate rules for confusion pairs that have reached threshold."""
        self._ensure_index()
        for pred_label, true_counts in self._confusion_counts.items():
            for true_label, count in true_counts.items():
                if count < CONFUSION_THRESHOLD:
                    continue
                pair = (pred_label, true_label)
                if pair in self._rule_generated:
                    continue
                # Need examples for both labels
                idxs_a = self._label_to_idxs.get(pred_label, [])
                idxs_b = self._label_to_idxs.get(true_label, [])
                if not idxs_a or not idxs_b:
                    self._rule_generated.add(pair)
                    continue
                # Get sample cases
                cases_a = []
                for i in idxs_a[:MAX_CASES_FOR_RULE]:
                    ex = self.examples[i]
                    q = ex.get("raw_question") or ex["input"]
                    cases_a.append(q[:200])
                cases_b = []
                for i in idxs_b[:MAX_CASES_FOR_RULE]:
                    ex = self.examples[i]
                    q = ex.get("raw_question") or ex["input"]
                    cases_b.append(q[:200])

                prompt = RULE_GEN_PROMPT.format(
                    label_a=pred_label,
                    label_b=true_label,
                    cases_a="\n".join(f"- {c}" for c in cases_a),
                    cases_b="\n".join(f"- {c}" for c in cases_b),
                )
                try:
                    response = self.call_llm(prompt)
                    rule = extract_json_field(response, "rule")
                    if rule:
                        self._rules[pair] = rule
                        # Also store reverse
                        self._rules[(true_label, pred_label)] = rule
                except Exception:
                    pass
                self._rule_generated.add(pair)

    def _find_relevant_rules(self, candidate_labels: set[str]) -> list[str]:
        """Find rules relevant to the candidate labels from retrieval."""
        relevant = []
        seen_pairs = set()
        for (a, b), rule in self._rules.items():
            canonical = tuple(sorted([a, b]))
            if canonical in seen_pairs:
                continue
            if a in candidate_labels or b in candidate_labels:
                relevant.append(f"• {a} vs {b}: {rule}")
                seen_pairs.add(canonical)
                if len(relevant) >= MAX_RULES_IN_PROMPT:
                    break
        return relevant

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        # Generate any pending rules before first prediction
        self._generate_pending_rules()

        idxs = self._select(input)
        if not idxs:
            prompt = PROMPT_NO_RULES.format(examples_section="", input=input)
            response = self.call_llm(prompt)
            answer = extract_json_field(response, "final_answer")
            return answer, {"full_response": response}

        # Build examples section
        parts = []
        total = 0
        candidate_labels: set[str] = set()
        for i in idxs:
            ex = self.examples[i]
            q = ex.get("raw_question", ex["input"])
            part = f"Q: {q}\nA: {ex['target']}"
            if total + len(part) > MAX_CHARS:
                break
            parts.append(part)
            candidate_labels.add(ex.get("target", ""))
            total += len(part) + 2
        examples_section = "\n\n".join(parts)

        # Find relevant discrimination rules
        relevant_rules = self._find_relevant_rules(candidate_labels)

        if relevant_rules:
            rules_section = "\n".join(relevant_rules)
            prompt = PROMPT_WITH_RULES.format(
                rules_section=rules_section,
                examples_section=examples_section,
                input=input,
            )
        else:
            prompt = PROMPT_NO_RULES.format(
                examples_section=examples_section,
                input=input,
            )

        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")
        return answer, {
            "full_response": response,
            "rules_shown": len(relevant_rules),
            "candidate_labels": list(candidate_labels),
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for r in batch_results:
            ex = {"input": str(r["input"]), "target": str(r["ground_truth"])}
            if "raw_question" in r:
                ex["raw_question"] = str(r["raw_question"])
            self.examples.append(ex)
            # Track confusion
            pred = r.get("prediction", "")
            true = str(r["ground_truth"])
            if pred and pred != true:
                self._confusion_counts[pred][true] += 1
        self._index_dirty = True

    def get_state(self) -> str:
        return json.dumps({
            "examples": self.examples,
            "confusion_counts": {k: dict(v) for k, v in self._confusion_counts.items()},
            "rules": {f"{a}|||{b}": r for (a, b), r in self._rules.items()},
            "rule_generated": [f"{a}|||{b}" for a, b in self._rule_generated],
        }, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = data.get("examples", [])
        self._confusion_counts = defaultdict(Counter)
        for k, v in data.get("confusion_counts", {}).items():
            self._confusion_counts[k] = Counter(v)
        self._rules = {}
        for key, rule in data.get("rules", {}).items():
            parts = key.split("|||", 1)
            if len(parts) == 2:
                self._rules[(parts[0], parts[1])] = rule
        self._rule_generated = set()
        for key in data.get("rule_generated", []):
            parts = key.split("|||", 1)
            if len(parts) == 2:
                self._rule_generated.add((parts[0], parts[1]))
        self._index_dirty = True
