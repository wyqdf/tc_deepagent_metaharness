"""BM25 retrieval with contrastive partner augmentation.

Mechanism change vs iter001_label_coverage_bm25: After selecting top-k examples
via BM25 + label-coverage, identifies "contrastive partners" — examples that are
lexically similar to the top retrievals but have a DIFFERENT target label. These
are shown in a separate prompt section to help the model discriminate between
confusable labels.

This addresses a key failure mode: the model confuses similar labels (e.g.,
"受贿" vs "利用影响力受贿", "假冒注册商标" vs "销售假冒注册商标的商品") because
it only sees positive examples for each label. Contrastive partners explicitly
show what distinguishes similar-but-different cases.
"""
# Candidate memory-system logic continues below.


from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from typing import Any

from harness.agent_protocol import BaseAgentMemory, extract_json_field

PROMPT_TEMPLATE = """Solve the problem below using both the closest matching examples AND the contrastive examples that show similar inputs with DIFFERENT answers.

**Closest matching examples:**
{examples_section}

{contrastive_section}
**Problem:**
{input}

**Instructions:**
- The "Closest matching examples" are the most relevant prior cases.
- The "Contrastive examples" are lexically similar to the above but have a DIFFERENT correct answer — use them to identify distinguishing features.
- Decide based on distinguishing features, not just surface similarity.
- Respond in JSON format.

{{"reasoning": "[step-by-step including which distinguishing features apply]", "final_answer": "[your answer]"}}"""

MAX_CHARS = 28000
TOP_K = 12
CANDIDATE_POOL = 64
NGRAM_MIN = 2
NGRAM_MAX = 3
ANCHORS = 3
PER_LABEL_CAP = 2
CONTRAST_PER_ANCHOR = 1
CONTRAST_CAP = 4


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


class ContrastiveRetrieval(BaseAgentMemory):
    """BM25 retrieval with contrastive partner augmentation."""

    def __init__(self, llm):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self._docs_tokens: list[list[str]] = []
        self._doc_tfs: list[Counter] = []
        self._doc_lens: list[int] = []
        self._idf: dict[str, float] = {}
        self._avgdl: float = 0.0
        self._partners: dict[int, int] = {}  # idx -> nearest different-label idx
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
        self._compute_partners()
        self._index_dirty = False

    def _compute_partners(self) -> None:
        """For each example, find nearest neighbor with a different target."""
        n = len(self.examples)
        self._partners = {}
        for i in range(n):
            target_i = self.examples[i].get("target", "")
            qtoks = self._docs_tokens[i]
            best_score = -1.0
            best_j = -1
            for j in range(n):
                if j == i or self.examples[j].get("target", "") == target_i:
                    continue
                s = _bm25_score(
                    qtoks, self._doc_tfs[j], self._doc_lens[j], self._avgdl, self._idf
                )
                if s > best_score:
                    best_score = s
                    best_j = j
            if best_j >= 0:
                self._partners[i] = best_j

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

        # Anchors: top by relevance
        for i in pool[:ANCHORS]:
            chosen.append(i)
            label_counts[labels[i]] += 1
            if len(chosen) >= TOP_K:
                return chosen

        # Coverage pass: one per unseen label
        for i in pool:
            if i in chosen:
                continue
            if label_counts[labels[i]] == 0:
                chosen.append(i)
                label_counts[labels[i]] += 1
                if len(chosen) >= TOP_K:
                    return chosen

        # Cap pass: fill up to PER_LABEL_CAP
        for i in pool:
            if i in chosen:
                continue
            if label_counts[labels[i]] < PER_LABEL_CAP:
                chosen.append(i)
                label_counts[labels[i]] += 1
                if len(chosen) >= TOP_K:
                    return chosen

        # Remainder
        for i in pool:
            if i in chosen:
                continue
            chosen.append(i)
            if len(chosen) >= TOP_K:
                break
        return chosen

    def _get_contrastive(self, chosen: list[int]) -> list[int]:
        """Get contrastive partners for the anchor examples."""
        chosen_set = set(chosen)
        contrast: list[int] = []
        seen_labels: set[str] = set()
        for i in chosen[:ANCHORS]:
            partner = self._partners.get(i)
            if partner is None or partner in chosen_set or partner in set(contrast):
                continue
            partner_label = self.examples[partner].get("target", "")
            if partner_label in seen_labels:
                continue
            contrast.append(partner)
            seen_labels.add(partner_label)
            if len(contrast) >= CONTRAST_CAP:
                break
        return contrast

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        idxs = self._select(input)
        contrast_idxs = self._get_contrastive(idxs)

        # Format primary examples
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

        # Format contrastive section
        contrastive_section = ""
        if contrast_idxs:
            c_parts = []
            for i in contrast_idxs:
                ex = self.examples[i]
                q = ex.get("raw_question", ex["input"])
                part = f"Q: {q}\nA: {ex['target']}"
                if total + len(part) > MAX_CHARS:
                    break
                c_parts.append(part)
                total += len(part) + 2
            if c_parts:
                contrastive_section = "**Contrastive examples (similar but DIFFERENT answer):**\n" + "\n\n".join(c_parts) + "\n\n"

        prompt = PROMPT_TEMPLATE.format(
            examples_section=examples_section,
            contrastive_section=contrastive_section,
            input=input,
        )
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")
        return answer, {"full_response": response, "num_examples": len(self.examples)}

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for r in batch_results:
            ex = {"input": str(r["input"]), "target": str(r["ground_truth"])}
            if "raw_question" in r:
                ex["raw_question"] = str(r["raw_question"])
            self.examples.append(ex)
        self._index_dirty = True

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = data.get("examples", [])
        self._index_dirty = True
