"""Self-test recipe memory with BM25 retrieval.

Mechanism change vs iter005_confusion_verify (base): Fundamentally different
MEMORY CONTENT approach. Instead of two-pass verification, enriches the stored
examples themselves with LLM-generated decision recipes.

After all training batches are ingested, at first predict call:
1. Performs leave-one-out self-tests on a sample of training examples
2. For examples the model gets WRONG via retrieval-based prediction,
   generates a short "recipe" (decision rule) explaining why the correct
   answer fits given the example's surface features
3. Recipes travel WITH their examples in the prompt as extra context

At predict time (single pass):
- BM25 retrieval + label-coverage selection (same as base)
- Examples that have recipes show them inline: "Q: ... A: ... Recipe: ..."
- The recipe provides discriminative guidance at exactly the point where
  the relevant example appears

Key differences from prior systems:
- iter007_inline_rules_retrieval: rules are PAIR-keyed (X vs Y), rendered
  separately from examples. Recipes here are EXAMPLE-keyed, travel with
  their specific example.
- iter005_confusion_verify: uses two LLM calls per prediction. This uses
  one call per prediction but invests LLM calls upfront during learning.
- iter004_error_notes_retrieval: notes are rendered in a separate section.
  Recipes are inline with their example, providing local context.

The single-pass design avoids verification corruption while still providing
disambiguation guidance. Recipes are generated ONLY for hard cases, focusing
prompt budget where it matters most.

Axes changed: MEMORY CONTENT (per-example recipes) + LEARNING STRATEGY
(self-test to identify hard cases) — genuinely new mechanism.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from typing import Any

from harness.agent_protocol import BaseAgentMemory, extract_json_field

PROMPT_TEMPLATE = """Solve the problem below based on the examples provided.

{examples_section}

**Problem:**
{input}

**Instructions:**
- The examples above are the most relevant prior cases.
- Some examples include a "Recipe:" line explaining why that answer is correct given its features. Use these to guide your reasoning.
- Follow the patterns shown in the examples.
- Respond in JSON format.

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""

SELFTEST_PROMPT = """Solve the problem below based on the examples provided.

{examples_section}

**Problem:**
{input}

**Instructions:**
- Follow the patterns shown in the examples.
- Respond in JSON format.

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""

RECIPE_PROMPT = """A model failed to identify the correct answer for this case. Write a SHORT decision rule (1-2 sentences, max 40 words) explaining why the correct answer fits.

PROBLEM:
{query}

CORRECT ANSWER: {answer}

MODEL'S WRONG PREDICTION: {wrong_pred}

Write a rule of the form: "If [specific features visible in the problem], the answer is [{answer}] because [distinguishing reason]."
Be concrete and grounded in the problem's actual content. Respond in JSON.

{{"recipe": "[the rule]"}}"""

MAX_CHARS = 28000
TOP_K = 14
CANDIDATE_POOL = 64
NGRAM_MIN = 2
NGRAM_MAX = 3
ANCHORS = 3
PER_LABEL_CAP = 2
SELF_TEST_N = 30
SELF_TEST_TOP_K = 10
MAX_RECIPES = 25


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


class SelfTestRecipeMemory(BaseAgentMemory):
    """BM25 retrieval with per-example LLM-written decision recipes for hard cases."""

    def __init__(self, llm):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self.recipes: dict[int, str] = {}  # example_idx -> recipe string
        self._docs_tokens: list[list[str]] = []
        self._doc_tfs: list[Counter] = []
        self._doc_lens: list[int] = []
        self._idf: dict[str, float] = {}
        self._avgdl: float = 0.0
        self._index_dirty = True
        self._recipes_generated = False

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
        self._index_dirty = False

    def _select(self, query: str, exclude: int = -1) -> list[int]:
        self._ensure_index()
        if not self.examples:
            return []
        qtoks = _tokenize(query)
        n = len(self.examples)
        scored = sorted(
            range(n),
            key=lambda i: -_bm25_score(
                qtoks, self._doc_tfs[i], self._doc_lens[i], self._avgdl, self._idf
            ),
        )
        pool = [i for i in scored if i != exclude][:CANDIDATE_POOL]
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

    def _generate_recipes(self) -> None:
        """Run self-tests and generate recipes for failures."""
        if self._recipes_generated or len(self.examples) < 10:
            return
        self._recipes_generated = True
        self._ensure_index()

        import random
        n = len(self.examples)
        sample_size = min(SELF_TEST_N, n)
        sample_idxs = random.sample(range(n), sample_size)

        failures: list[tuple[int, str]] = []  # (idx, wrong_prediction)
        for idx in sample_idxs:
            ex = self.examples[idx]
            query = ex.get("raw_question") or ex["input"]
            # Leave-one-out retrieval
            sel = self._select(query, exclude=idx)[:SELF_TEST_TOP_K]
            if not sel:
                continue
            parts = []
            total = 0
            for i in sel:
                e = self.examples[i]
                q = e.get("raw_question", e["input"])
                part = f"Q: {q}\nA: {e['target']}"
                if total + len(part) > MAX_CHARS:
                    break
                parts.append(part)
                total += len(part) + 2
            if not parts:
                continue
            prompt = SELFTEST_PROMPT.format(
                examples_section="\n\n".join(parts), input=query
            )
            response = self.call_llm(prompt)
            pred = extract_json_field(response, "final_answer")
            if pred and pred != ex["target"]:
                failures.append((idx, pred))

        # Generate recipes for failures (capped)
        for idx, wrong_pred in failures[:MAX_RECIPES]:
            if idx in self.recipes:
                continue
            ex = self.examples[idx]
            query = ex.get("raw_question") or ex["input"]
            prompt = RECIPE_PROMPT.format(
                query=query[:2000],
                answer=ex["target"],
                wrong_pred=wrong_pred,
            )
            response = self.call_llm(prompt)
            recipe = extract_json_field(response, "recipe")
            if recipe:
                self.recipes[idx] = recipe

    def _format_examples(self, query: str) -> str:
        idxs = self._select(query)
        if not idxs:
            return ""
        parts = []
        total = 0
        for i in idxs:
            ex = self.examples[i]
            q = ex.get("raw_question", ex["input"])
            part = f"Q: {q}\nA: {ex['target']}"
            if i in self.recipes:
                part += f"\nRecipe: {self.recipes[i]}"
            if total + len(part) > MAX_CHARS:
                break
            parts.append(part)
            total += len(part) + 2
        return "\n\n".join(parts)

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        # Generate recipes on first predict (lazy)
        self._generate_recipes()
        examples_section = self._format_examples(input)
        prompt = PROMPT_TEMPLATE.format(examples_section=examples_section, input=input)
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")
        return answer, {"full_response": response, "num_recipes": len(self.recipes)}

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for r in batch_results:
            ex = {"input": str(r["input"]), "target": str(r["ground_truth"])}
            if "raw_question" in r:
                ex["raw_question"] = str(r["raw_question"])
            self.examples.append(ex)
        self._index_dirty = True
        self._recipes_generated = False

    def get_state(self) -> str:
        return json.dumps({
            "examples": self.examples,
            "recipes": {str(k): v for k, v in self.recipes.items()},
        }, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = data.get("examples", [])
        self.recipes = {int(k): v for k, v in data.get("recipes", {}).items()}
        self._index_dirty = True
        self._recipes_generated = bool(self.recipes)
