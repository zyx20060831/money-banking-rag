from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from .models import Chunk, Evidence
from .text_utils import (
    normalize_for_search,
    query_expansion_phrases,
    query_negative_context_phrases,
    tokenize,
)


DEFAULT_TOP_K = 8

_EXPANSION_TERM_WEIGHT = 0.55
_EXPANSION_SECTION_BONUS = 4.0
_EXPANSION_LEADING_TEXT_BONUS = 3.0
_MIN_ORIGINAL_TERM_IDF_FOR_BONUS = 1.0
_NEGATIVE_CONTEXT_MULTIPLIER = 0.55
_TABLE_OF_CONTENTS_MULTIPLIER = 0.20


class DocumentNotFoundError(ValueError):
    pass


def load_manifest(data_dir: Path) -> dict[str, Any]:
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"未找到 {manifest_path}，请先运行 build 命令提取 PDF。"
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def resolve_document(manifest: dict[str, Any], selector: str) -> dict[str, Any]:
    documents = manifest.get("documents", [])
    exact = [
        doc
        for doc in documents
        if selector.casefold()
        in {
            doc["document_id"].casefold(),
            doc["source_name"].casefold(),
        }
    ]
    if len(exact) == 1:
        return exact[0]

    partial = [
        doc
        for doc in documents
        if selector.casefold() in doc["document_id"].casefold()
        or selector.casefold() in doc["source_name"].casefold()
    ]
    if len(partial) == 1:
        return partial[0]

    names = "、".join(doc["source_name"] for doc in documents) or "（无）"
    if not partial:
        raise DocumentNotFoundError(f"找不到文档“{selector}”。可选文档：{names}")
    raise DocumentNotFoundError(
        f"文档选择“{selector}”不唯一，请输入更完整的名称。可选文档：{names}"
    )


def load_chunks(document: dict[str, Any]) -> list[Chunk]:
    chunks: list[Chunk] = []
    with Path(document["chunks_path"]).open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                chunks.append(Chunk.from_dict(json.loads(line)))
    return chunks


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], *, k1: float = 1.5, b: float = 0.75):
        if not chunks:
            raise ValueError("文档没有可检索文本块")
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.term_frequencies = [Counter(tokenize(chunk.text)) for chunk in chunks]
        self.lengths = [sum(counter.values()) for counter in self.term_frequencies]
        self.average_length = sum(self.lengths) / len(self.lengths)

        self.document_frequency: Counter[str] = Counter()
        for counter in self.term_frequencies:
            self.document_frequency.update(counter.keys())

    def _idf(self, token: str) -> float:
        count = self.document_frequency.get(token, 0)
        total = len(self.chunks)
        return math.log(1.0 + (total - count + 0.5) / (count + 0.5))

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[Evidence]:
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        original_query_terms = Counter(tokenize(query))
        query_terms = Counter(original_query_terms)
        query_weights = {token: 1.0 for token in query_terms}
        expansion_phrases = query_expansion_phrases(query)
        negative_context_phrases = query_negative_context_phrases(query)
        expansion_text = " ".join(expansion_phrases)
        for token in set(tokenize(expansion_text)):
            if token in query_terms:
                continue
            query_terms[token] = 1
            query_weights[token] = _EXPANSION_TERM_WEIGHT
        scored: list[Evidence] = []
        for index, (chunk, frequencies, length) in enumerate(
            zip(self.chunks, self.term_frequencies, self.lengths, strict=True)
        ):
            score = 0.0
            normalization = self.k1 * (
                1 - self.b + self.b * length / max(self.average_length, 1.0)
            )
            for token, query_count in query_terms.items():
                frequency = frequencies.get(token, 0)
                if not frequency:
                    continue
                term_score = self._idf(token) * (
                    frequency * (self.k1 + 1) / (frequency + normalization)
                )
                score += (
                    term_score
                    * query_weights[token]
                    * (1.0 + 0.15 * (query_count - 1))
                )

            section_terms = set(tokenize(chunk.section))
            score += 0.25 * sum(
                self._idf(term) * query_weights[term]
                for term in query_terms
                if term in section_terms
            )

            matched_distinctive_original_term = any(
                frequencies.get(term, 0) > 0
                and self._idf(term) >= _MIN_ORIGINAL_TERM_IDF_FOR_BONUS
                for term in original_query_terms
            )
            if matched_distinctive_original_term:
                normalized_section = normalize_for_search(chunk.section)
                normalized_leading_text = normalize_for_search(chunk.text[:120])
                phrase_bonus = 0.0
                for phrase in expansion_phrases:
                    normalized_phrase = normalize_for_search(phrase)
                    in_section = normalized_phrase in normalized_section
                    in_leading_text = normalized_phrase in normalized_leading_text
                    if not in_section and not in_leading_text:
                        continue
                    phrase_terms = set(tokenize(phrase))
                    if not phrase_terms:
                        continue
                    average_idf = sum(
                        self._idf(term) for term in phrase_terms
                    ) / len(phrase_terms)
                    multiplier = (
                        _EXPANSION_SECTION_BONUS
                        if in_section
                        else _EXPANSION_LEADING_TEXT_BONUS
                    )
                    phrase_bonus = max(
                        phrase_bonus,
                        multiplier * average_idf * _EXPANSION_TERM_WEIGHT,
                    )
                score += phrase_bonus

            if negative_context_phrases:
                normalized_chunk = normalize_for_search(chunk.text)
                if any(
                    normalize_for_search(phrase) in normalized_chunk
                    for phrase in negative_context_phrases
                ):
                    score *= _NEGATIVE_CONTEXT_MULTIPLIER

            if normalize_for_search(chunk.text[:20]).startswith("目录"):
                score *= _TABLE_OF_CONTENTS_MULTIPLIER
            if score > 0:
                scored.append(Evidence(chunk=chunk, score=score))

        scored.sort(key=lambda evidence: evidence.score, reverse=True)

        selected: list[Evidence] = []
        per_page: Counter[int] = Counter()
        for evidence in scored:
            page = evidence.chunk.page
            if per_page[page] >= 2:
                continue
            selected.append(evidence)
            per_page[page] += 1
            if len(selected) >= top_k:
                break
        return selected
