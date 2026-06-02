"""BM25 retrieval with confusion-pair awareness.

Mechanism change vs fewshot_all: Uses BM25 retrieval like label_coverage_bm25,
but adds a learning strategy that tracks which label pairs the model confuses.
During selection, when a retrieved label has a known confusion partner, the
system ensures examples of BOTH labels appear in the prompt so the model can
discriminate between them.

This addresses a different failure mode: even with good retrieval, the model
may consistently confuse similar labels (e.g. "pneumonia" vs "bronchial asthma",
"诈骗" vs "合同诈骗"). By showing contrastive examples of both confused labels,
the model can learn the distinguishing features.
"""
# Candidate memory-system logic continues below.


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
- Pay close attention to differences between similar-looking answers in the examples.
- Follow the patterns shown in the examples.
- Respond in JSON format.

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""

MAX_CHARS = 30000
TOP_K = 16
CANDIDATE_POOL = 64
NGRAM_MIN = 2
NGRAM_MAX = 3


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


class ConfusionAwareBm25(BaseAgentMemory):
    """BM25 retrieval with confusion-pair contrastive selection."""

    def __init__(self, llm):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        # confusion_pairs: maps label -> set of labels it gets confused with
        self.confusion_partners: dict[str, Counter] = defaultdict(Counter)
        self._docs_tokens: list[list[str]] = []
        self._doc_tfs: list[Counter] = []
        self._doc_lens: list[int] = []
        self._idf: dict[str, float] = {}
        self._avgdl: float = 0.0
        self._index_dirty = True

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
        if not pool:
            return []

        labels = [self.examples[i].get("target", "") for i in range(n)]
        chosen: list[int] = []
        chosen_labels: set[str] = set()

        # Phase 1: top 3 anchors by relevance
        for i in pool[:3]:
            chosen.append(i)
            chosen_labels.add(labels[i])
            if len(chosen) >= TOP_K:
                return chosen

        # Phase 2: for each anchor label, find confusion partners and include them
        needed_labels: set[str] = set()
        for lbl in list(chosen_labels):
            if lbl in self.confusion_partners:
                # Add top confusion partners
                for partner, _cnt in self.confusion_partners[lbl].most_common(2):
                    if partner not in chosen_labels:
                        needed_labels.add(partner)

        # Find best examples for needed labels from pool
        for i in pool:
            if i in chosen:
                continue
            if labels[i] in needed_labels:
                chosen.append(i)
                chosen_labels.add(labels[i])
                needed_labels.discard(labels[i])
                if len(chosen) >= TOP_K:
                    return chosen

        # Phase 3: label coverage for remaining slots
        for i in pool:
            if i in chosen:
                continue
            if labels[i] not in chosen_labels:
                chosen.append(i)
                chosen_labels.add(labels[i])
                if len(chosen) >= TOP_K:
                    return chosen

        # Phase 4: fill remaining
        for i in pool:
            if i in chosen:
                continue
            chosen.append(i)
            if len(chosen) >= TOP_K:
                break
        return chosen

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
            if total + len(part) > MAX_CHARS:
                break
            parts.append(part)
            total += len(part) + 2
        return "\n\n".join(parts)

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        examples_section = self._format_examples(input)
        prompt = PROMPT_TEMPLATE.format(examples_section=examples_section, input=input)
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")
        return answer, {"full_response": response, "num_examples": len(self.examples)}

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for r in batch_results:
            ex = {"input": str(r["input"]), "target": str(r["ground_truth"])}
            if "raw_question" in r:
                ex["raw_question"] = str(r["raw_question"])
            self.examples.append(ex)

            # Track confusion pairs from errors
            if not r.get("was_correct", True) and r.get("prediction"):
                pred = str(r["prediction"]).strip()
                true = str(r["ground_truth"]).strip()
                if pred and true and pred != true:
                    self.confusion_partners[true][pred] += 1
                    self.confusion_partners[pred][true] += 1
        self._index_dirty = True

    def get_state(self) -> str:
        # Serialize confusion_partners as dict of dicts
        cp = {k: dict(v) for k, v in self.confusion_partners.items()}
        return json.dumps(
            {"examples": self.examples, "confusion_partners": cp},
            ensure_ascii=False,
        )

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = data.get("examples", [])
        cp = data.get("confusion_partners", {})
        self.confusion_partners = defaultdict(Counter)
        for k, v in cp.items():
            self.confusion_partners[k] = Counter(v)
        self._index_dirty = True
