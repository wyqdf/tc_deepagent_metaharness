"""BM25 retrieval + progressive error notes from actual confusion pairs.

Mechanism change vs iter001_label_coverage_bm25 (base): Adds a LEARNING mechanism.
During learn_from_batch, tracks confusion pairs (predicted->true). When a pair
recurs 2+ times, generates a short discriminative note via LLM. At predict time,
notes relevant to the retrieved labels are injected as a preamble.

This addresses a key weakness: the model repeatedly confuses similar labels
(e.g., "盗窃" vs "破坏电力设备", "受贿" vs "利用影响力受贿") because retrieval
alone doesn't teach it the distinguishing features. Error notes provide explicit
discrimination rules learned from actual mistakes.

Cost: ~1 LLM call per new recurring confusion pair (typically 3-10 total).
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

{notes_section}{examples_section}

**Problem:**
{input}

**Instructions:**
- The notes (if any) describe how to distinguish answers the model has previously confused. Apply them.
- The examples are the most relevant prior cases.
- Respond in JSON format.

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""

NOTE_PROMPT = """The model repeatedly predicted "{predicted}" when the true answer was "{true}".

Example inputs where this confusion occurred:
{error_inputs}

Cases with the TRUE answer ("{true}"):
{true_cases}

Cases with the WRONG predicted answer ("{predicted}"):
{wrong_cases}

Write a SHORT rule (<=50 words) to distinguish "{true}" from "{predicted}". Focus on key differentiating features, keywords, or patterns. Respond in JSON.

{{"note": "[the rule]"}}"""

MAX_CHARS = 30000
TOP_K = 16
CANDIDATE_POOL = 64
NGRAM_MIN = 2
NGRAM_MAX = 3
ANCHORS = 3
PER_LABEL_CAP = 2
NOTE_BUDGET = 4000
CONFUSION_THRESHOLD = 2


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


class ErrorNotesRetrieval(BaseAgentMemory):
    """BM25 retrieval with label-coverage + progressive error notes."""

    def __init__(self, llm):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self.notes: list[dict[str, str]] = []
        self._docs_tokens: list[list[str]] = []
        self._doc_tfs: list[Counter] = []
        self._doc_lens: list[int] = []
        self._idf: dict[str, float] = {}
        self._avgdl: float = 0.0
        self._index_dirty = True
        self._confusion_counts: Counter = Counter()
        self._confusion_inputs: dict[tuple[str, str], list[str]] = defaultdict(list)
        self._noted_pairs: set[tuple[str, str]] = set()

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

    def _format_notes(self, retrieved_labels: list[str]) -> str:
        if not self.notes:
            return ""
        label_set = set(retrieved_labels)
        relevant = []
        for note in self.notes:
            if note["predicted"] in label_set or note["true"] in label_set:
                relevant.append(note)
        if not relevant:
            return ""
        lines = []
        budget = NOTE_BUDGET
        for note in relevant:
            line = f'- When choosing between "{note["predicted"]}" and "{note["true"]}": {note["note"]}'
            if len(line) > budget:
                break
            lines.append(line)
            budget -= len(line)
        if not lines:
            return ""
        return "**Learned distinctions (from past errors):**\n" + "\n".join(lines) + "\n\n"

    def _generate_note(self, pair: tuple[str, str]) -> None:
        predicted, true = pair
        error_inputs = self._confusion_inputs[pair][:3]
        true_cases = [
            (ex.get("raw_question") or ex["input"])[:300]
            for ex in self.examples if ex.get("target") == true
        ][:3]
        wrong_cases = [
            (ex.get("raw_question") or ex["input"])[:300]
            for ex in self.examples if ex.get("target") == predicted
        ][:3]
        if not true_cases:
            return
        prompt = NOTE_PROMPT.format(
            predicted=predicted,
            true=true,
            error_inputs="\n---\n".join(f"INPUT: {inp[:300]}" for inp in error_inputs),
            true_cases="\n---\n".join(f"CASE: {c}" for c in true_cases),
            wrong_cases="\n---\n".join(f"CASE: {c}" for c in wrong_cases) if wrong_cases else "(none stored)",
        )
        response = self.call_llm(prompt)
        note_text = extract_json_field(response, "note")
        if note_text and len(note_text) > 5:
            self.notes.append({"predicted": predicted, "true": true, "note": note_text})
            self._noted_pairs.add(pair)

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        idxs = self._select(input)
        retrieved_labels = [self.examples[i].get("target", "") for i in idxs] if idxs else []
        notes_section = self._format_notes(retrieved_labels)
        budget = MAX_CHARS - len(notes_section)
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
        examples_section = "\n\n".join(parts)
        prompt = PROMPT_TEMPLATE.format(
            notes_section=notes_section, examples_section=examples_section, input=input
        )
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")
        return answer, {"full_response": response, "num_notes": len(self.notes)}

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for r in batch_results:
            ex = {"input": str(r["input"]), "target": str(r["ground_truth"])}
            if "raw_question" in r:
                ex["raw_question"] = str(r["raw_question"])
            self.examples.append(ex)
            # Track confusion pairs
            if not r.get("was_correct", True) and r.get("prediction"):
                pred = str(r["prediction"]).strip()
                true = str(r["ground_truth"]).strip()
                if pred and true and pred != true:
                    pair = (pred, true)
                    self._confusion_counts[pair] += 1
                    self._confusion_inputs[pair].append(
                        str(r.get("raw_question") or r["input"])[:300]
                    )
                    if (self._confusion_counts[pair] >= CONFUSION_THRESHOLD
                            and pair not in self._noted_pairs):
                        self._ensure_index()
                        self._generate_note(pair)
        self._index_dirty = True

    def get_state(self) -> str:
        return json.dumps({
            "examples": self.examples,
            "notes": self.notes,
            "confusion_counts": {f"{k[0]}|||{k[1]}": v for k, v in self._confusion_counts.items()},
            "confusion_inputs": {f"{k[0]}|||{k[1]}": v for k, v in self._confusion_inputs.items()},
        }, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = data.get("examples", [])
        self.notes = data.get("notes", [])
        self._confusion_counts = Counter()
        self._confusion_inputs = defaultdict(list)
        self._noted_pairs = set()
        for k, v in data.get("confusion_counts", {}).items():
            parts = k.split("|||", 1)
            if len(parts) == 2:
                self._confusion_counts[(parts[0], parts[1])] = v
        for k, v in data.get("confusion_inputs", {}).items():
            parts = k.split("|||", 1)
            if len(parts) == 2:
                self._confusion_inputs[(parts[0], parts[1])] = v
        for note in self.notes:
            self._noted_pairs.add((note["predicted"], note["true"]))
        self._index_dirty = True
