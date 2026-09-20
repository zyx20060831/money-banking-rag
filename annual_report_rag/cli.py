from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .extractor import build_corpus
from .llm import LLMConfig
from .retriever import (
    DEFAULT_TOP_K,
    DocumentNotFoundError,
    load_manifest,
)
from .service import AnnualReportQA, format_result_markdown, parse_questions


def _add_retrieval_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="索引目录")
    parser.add_argument(
        "--document",
        required=True,
        help="文档 ID、完整文件名或唯一文件名片段，例如“宁波银行”",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"返回的证据块数量（默认 {DEFAULT_TOP_K}）",
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="annual-report-rag",
        description="使用 pdfplumber、BM25 和可配置大模型完成银行年报及货币政策报告问答。",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="提取 PDF 并建立文本索引")
    build_parser.add_argument("--pdf-dir", type=Path, default=Path("."))
    build_parser.add_argument("--data-dir", type=Path, default=Path("data"))
    build_parser.add_argument("--chunk-size", type=int, default=800)
    build_parser.add_argument("--overlap", type=int, default=120)
    build_parser.add_argument("--margin-lines", type=int, default=2)
    build_parser.add_argument("--force", action="store_true", help="强制重新提取")

    subparsers.add_parser("documents", help="列出已建立索引的文档").add_argument(
        "--data-dir", type=Path, default=Path("data")
    )

    search_parser = subparsers.add_parser("search", help="只检索并展示参考原文")
    _add_retrieval_arguments(search_parser)
    search_parser.add_argument("--query", required=True, help="检索问题")

    ask_parser = subparsers.add_parser("ask", help="检索后调用大模型回答")
    _add_retrieval_arguments(ask_parser)
    ask_parser.add_argument("--query", required=True, help="问题")

    batch_parser = subparsers.add_parser("batch", help="批量运行编号题库")
    _add_retrieval_arguments(batch_parser)
    batch_parser.add_argument(
        "--questions", type=Path, default=Path("annual_report_questions.txt")
    )
    batch_parser.add_argument(
        "--output", type=Path, default=Path("output/answers.md")
    )
    return parser


def _run_build(args: argparse.Namespace) -> int:
    documents = build_corpus(
        args.pdf_dir,
        args.data_dir,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        margin_lines=args.margin_lines,
        force=args.force,
    )
    print(f"索引完成，共 {len(documents)} 份文档。")
    print("说明：build 只建立索引，不会回答问题。")
    print("下一步可运行：python -m annual_report_rag documents")
    print(
        '提问示例：python -m annual_report_rag ask '
        '--document "宁波银行" --query "银行的净息差和资本充足率分别是多少？"'
    )
    return 0


def _run_documents(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.data_dir)
    for doc in manifest.get("documents", []):
        coverage = doc.get("text_layer_coverage")
        coverage_line = (
            f"，可提取文本页占比：{coverage:.1%}" if coverage is not None else ""
        )
        print(
            f"{doc['document_id']}\n"
            f"  文件：{doc['source_name']}\n"
            f"  页数：{doc['page_count']}，文本块：{doc['chunk_count']}"
            f"{coverage_line}"
        )
    return 0


def _make_service(args: argparse.Namespace) -> AnnualReportQA:
    return AnnualReportQA(args.data_dir, args.document)


def _run_search(args: argparse.Namespace) -> int:
    service = _make_service(args)
    result = service.ask(args.query, top_k=args.top_k, llm_config=None)
    result.answer = "以下为按相关性排序的原文，尚未调用大模型生成回答。"
    print(format_result_markdown(result, heading_level=2))
    return 0


def _run_ask(args: argparse.Namespace) -> int:
    service = _make_service(args)
    result = service.ask(
        args.query,
        top_k=args.top_k,
        llm_config=LLMConfig.from_environment(),
    )
    print(format_result_markdown(result, heading_level=2))
    return 0


def _run_batch(args: argparse.Namespace) -> int:
    service = _make_service(args)
    questions = parse_questions(args.questions)
    llm_config = LLMConfig.from_environment()
    parts = [
        f"# 文档问答：{service.document['source_name']}",
        "",
        f"题目数量：{len(questions)}",
        "",
    ]
    for position, (number, question) in enumerate(questions, start=1):
        print(f"[{position}/{len(questions)}] {question}")
        result = service.ask(question, top_k=args.top_k, llm_config=llm_config)
        result.question = f"{number}. {question}"
        parts.append(format_result_markdown(result, heading_level=2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(parts), encoding="utf-8", newline="\n")
    print(f"结果已写入：{args.output.resolve()}")
    if llm_config is None:
        print("提示：未设置 RAG_MODEL，本次报告仅包含检索结果。")
    return 0


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(errors="replace")
    parser = make_parser()
    args = parser.parse_args(argv)
    handlers = {
        "build": _run_build,
        "documents": _run_documents,
        "search": _run_search,
        "ask": _run_ask,
        "batch": _run_batch,
    }
    try:
        exit_code = handlers[args.command](args)
    except (FileNotFoundError, ValueError, DocumentNotFoundError, RuntimeError) as exc:
        parser.exit(2, f"错误：{exc}\n")
    sys.exit(exit_code)
