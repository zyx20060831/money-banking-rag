from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .llm import LLMConfig, answer_with_llm
from .models import Evidence
from .retriever import (
    DEFAULT_TOP_K,
    BM25Retriever,
    load_chunks,
    load_manifest,
    resolve_document,
)
from .text_utils import compact_excerpt


@dataclass(slots=True)
class QAResult:
    question: str
    document_name: str
    answer: str
    evidence: list[Evidence]


class AnnualReportQA:
    def __init__(self, data_dir: Path, document_selector: str):
        manifest = load_manifest(data_dir)
        self.document = resolve_document(manifest, document_selector)
        self.retriever = BM25Retriever(load_chunks(self.document))

    def search(self, question: str, top_k: int = DEFAULT_TOP_K) -> list[Evidence]:
        return self.retriever.search(question, top_k=top_k)

    def ask(
        self,
        question: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        llm_config: LLMConfig | None = None,
    ) -> QAResult:
        evidence = self.search(question, top_k=top_k)
        if llm_config is None:
            answer = (
                "未配置大模型，本次仅完成检索。请设置 RAG_MODEL（以及服务所需的 "
                "RAG_API_KEY、RAG_BASE_URL）后重新运行，以生成综合回答。"
            )
        else:
            answer = answer_with_llm(question, evidence, llm_config)
        return QAResult(
            question=question,
            document_name=self.document["source_name"],
            answer=answer,
            evidence=evidence,
        )


def parse_questions(path: Path) -> list[tuple[int, str]]:
    questions: list[tuple[int, str]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip() == "---":
            break
        match = re.match(r"^\s*(\d+)[.、]\s*(.+?)\s*$", line)
        if match:
            questions.append((int(match.group(1)), match.group(2)))
    if not questions:
        raise ValueError(f"题库 {path} 中没有找到编号问题")
    return questions


def format_result_markdown(result: QAResult, *, heading_level: int = 2) -> str:
    heading = "#" * heading_level
    lines = [
        f"{heading} {result.question}",
        "",
        f"**文档：** {result.document_name}",
        "",
        result.answer,
        "",
        f"{heading}# 参考原文",
        "",
    ]
    if not result.evidence:
        lines.append("未检索到相关段落。")
    else:
        for index, item in enumerate(result.evidence, start=1):
            chunk = item.chunk
            lines.extend(
                [
                    f"- **[E{index}] PDF 第 {chunk.page} 页｜{chunk.section}** "
                    f"（BM25={item.score:.3f}）",
                    "",
                    f"  > {compact_excerpt(chunk.text).replace(chr(10), ' ')}",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"
