"""Keyword inventory + MMR retrieval (no verification pass).

Exploration: fundamentally different prompt organization strategy.

Instead of the two-pass predict+verify approach used by all top systems,
this candidate uses a SINGLE-PASS approach with enriched context:

1. LABEL KEYWORD INVENTORY: At training time, extracts discriminative
   keywords for each label that appears 2+ times. Uses TF-IDF-like scoring
   to find terms that are frequent within a label but rare across other
   labels. This creates a compact "cheat sheet" of what distinguishes
   each label, presented at the top of the prompt.

2. MMR RETRIEVAL: Same proven MMR-diversified retrieval as iter012.

3. SINGLE PASS with inventory: The prompt shows the keyword inventory
   ABOVE the examples, giving the model abstract category knowledge
   before seeing concrete evidence. This is fundamentally different from
   verification (which uses 2 LLM calls) — it uses 1 LLM call with
   richer context.

Why this might beat two-pass verify:
- Verification can FLIP correct answers to wrong (observed in iter014)
- Single pass avoids this risk while still providing disambiguation signal
- The keyword inventory is zero-cost (no LLM calls at training time)
- On USPTO (unique targets), no inventory is built, so it degrades to
  pure MMR retrieval (same as iter012 pass1 without verify overhead)
- On LawBench/S2D, the inventory provides explicit label descriptions
  that help the model distinguish similar categories

Cost: 1 LLM call per prediction (vs 2 for verify systems).
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from typing import Any

from harness.agent_protocol import BaseAgentMemory, extract_json_field

PROMPT_TEMPLATE = """Solve the problem below based on the label reference and examples provided.

{inventory_section}**Retrieved examples:**
{examples_section}

**Problem:**
{input}

**Instructions:**
- Use the label reference (if shown) to understand what distinguishes each possible answer.
- Use the retrieved examples as concrete evidence for pattern matching.
- Pick the answer that best matches both the reference keywords and the example patterns.
- Respond in JSON format.

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""

MAX_CHARS = 28000
INVENTORY_BUDGET = 4000
TOP_K = 14
CANDIDATE_POOL = 64
NGRAM_MIN = 2
NGRAM_MAX = 3
MMR_LAMBDA = 0.7
MIN_EXAMPLES_PER_LABEL = 2
MAX_KEYWORDS_PER_LABEL = 8


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


class KeywordInventoryRetrieval(BaseAgentMemory):
    """Single-pass MMR retrieval with label keyword inventory."""

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
        self._inventory: dict[str, list[str]] = {}  # label -> keywords
        self._inventory_dirty = True

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

    def _build_inventory(self) -> None:
        """Extract discriminative keywords per label (no LLM call)."""
        if not self._inventory_dirty:
            return
        self._ensure_index()
        n_total = len(self.examples)
        if n_total == 0:
            self._inventory = {}
            self._inventory_dirty = False
            return

        # Use word-level tokens for readable keywords
        label_word_sets: dict[str, list[set[str]]] = defaultdict(list)
        for i, ex in enumerate(self.examples):
            label = ex.get("target", "")
            text = ex.get("raw_question") or ex["input"]
            words = set(re.findall(r"[a-z0-9\u4e00-\u9fff]{2,}", text.lower()))
            label_word_sets[label].append(words)

        # Global document frequency of words
        global_df: Counter = Counter()
        for i, ex in enumerate(self.examples):
            text = ex.get("raw_question") or ex["input"]
            words = set(re.findall(r"[a-z0-9\u4e00-\u9fff]{2,}", text.lower()))
            for w in words:
                global_df[w] += 1

        self._inventory = {}
        for label, word_sets in label_word_sets.items():
            if len(word_sets) < MIN_EXAMPLES_PER_LABEL:
                continue
            # Term frequency within this label's examples
            label_tf: Counter = Counter()
            for ws in word_sets:
                for w in ws:
                    label_tf[w] += 1

            n_label = len(word_sets)
            scored_terms = []
            for term, count in label_tf.items():
                # Skip terms that appear in almost all docs
                if global_df[term] >= n_total * 0.6:
                    continue
                # Skip very short terms (likely noise)
                if len(term) < 2:
                    continue
                # Discriminative score
                label_freq = count / n_label
                other_count = global_df[term] - count
                other_freq = other_count / max(1, n_total - n_label)
                disc_score = label_freq * (1.0 - other_freq)
                if disc_score > 0.1:
                    scored_terms.append((disc_score, term))

            scored_terms.sort(reverse=True)
            keywords = [t for _, t in scored_terms[:MAX_KEYWORDS_PER_LABEL]]
            if keywords:
                self._inventory[label] = keywords

        self._inventory_dirty = False

    def _format_inventory(self, candidate_labels: set[str]) -> str:
        """Format keyword inventory for candidate labels."""
        if not self._inventory:
            return ""
        # Only show inventory for labels that appear in retrieved examples
        relevant = {l: kws for l, kws in self._inventory.items() if l in candidate_labels}
        if not relevant:
            return ""
        lines = ["**Label reference (distinguishing keywords):**"]
        total_chars = 0
        for label in sorted(relevant.keys()):
            kws = relevant[label]
            line = f"- {label}: {', '.join(kws)}"
            if total_chars + len(line) > INVENTORY_BUDGET:
                break
            lines.append(line)
            total_chars += len(line) + 1
        return "\n".join(lines) + "\n\n"

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
        pool_scores = [bm25_scores[i] for i in pool_indices]
        max_score = max(pool_scores) if pool_scores else 1.0
        if max_score <= 0:
            max_score = 1.0
        norm_scores = {i: bm25_scores[i] / max_score for i in pool_indices}
        chosen: list[int] = []
        chosen_set: set[int] = set()
        for _ in range(min(TOP_K, len(pool_indices))):
            best_idx = -1
            best_mmr = -float('inf')
            for i in pool_indices:
                if i in chosen_set:
                    continue
                relevance = norm_scores[i]
                if not chosen:
                    redundancy = 0.0
                else:
                    redundancy = max(
                        _jaccard_sim(self._doc_token_sets[i], self._doc_token_sets[j])
                        for j in chosen
                    )
                mmr = MMR_LAMBDA * relevance - (1 - MMR_LAMBDA) * redundancy
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i
            if best_idx < 0:
                break
            chosen.append(best_idx)
            chosen_set.add(best_idx)
        return chosen

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        self._build_inventory()
        idxs = self._mmr_select(input)

        if not idxs:
            prompt = PROMPT_TEMPLATE.format(
                inventory_section="", examples_section="(no examples yet)", input=input
            )
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

        # Build inventory section (only for labels in retrieved examples)
        inventory_section = self._format_inventory(candidate_labels)

        prompt = PROMPT_TEMPLATE.format(
            inventory_section=inventory_section,
            examples_section=examples_section,
            input=input,
        )
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")
        return answer, {"full_response": response, "inventory_labels": len(candidate_labels)}

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for r in batch_results:
            ex = {"input": str(r["input"]), "target": str(r["ground_truth"])}
            if "raw_question" in r:
                ex["raw_question"] = str(r["raw_question"])
            self.examples.append(ex)
        self._index_dirty = True
        self._inventory_dirty = True

    def get_state(self) -> str:
        return json.dumps({
            "examples": self.examples,
            "inventory": self._inventory,
        }, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = data.get("examples", [])
        self._inventory = data.get("inventory", {})
        self._index_dirty = True
        self._inventory_dirty = not bool(self._inventory)
