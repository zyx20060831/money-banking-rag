from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pdfplumber

from .models import Chunk
from .ocr import PageOCR
from .text_utils import (
    clean_page_lines,
    detect_section,
    find_repeated_margin_keys,
    section_level,
)


EXTRACTOR_SCHEMA_VERSION = 3


def make_document_id(path: Path) -> str:
    safe_stem = re.sub(r"[^0-9A-Za-z一-鿿_-]+", "_", path.stem).strip("_")
    digest = hashlib.sha1(path.name.encode("utf-8")).hexdigest()[:8]
    return f"{safe_stem}_{digest}"


def _split_lines_with_overlap(
    lines: list[str], chunk_size: int, overlap: int
) -> list[str]:
    if not lines:
        return []

    chunks: list[str] = []
    buffer: list[str] = []
    buffer_length = 0

    for line in lines:
        added_length = len(line) + (1 if buffer else 0)
        if buffer and buffer_length + added_length > chunk_size:
            chunks.append("\n".join(buffer))

            tail: list[str] = []
            tail_length = 0
            for previous_line in reversed(buffer):
                candidate_length = len(previous_line) + (1 if tail else 0)
                if tail and tail_length + candidate_length > overlap:
                    break
                tail.insert(0, previous_line)
                tail_length += candidate_length
                if tail_length >= overlap:
                    break
            buffer = tail
            buffer_length = tail_length

        buffer.append(line)
        buffer_length += len(line) + (1 if len(buffer) > 1 else 0)

    if buffer:
        chunks.append("\n".join(buffer))
    return chunks


def _chunk_page(
    *,
    lines: list[str],
    document_id: str,
    source_name: str,
    page_number: int,
    initial_section_path: list[tuple[int, str]],
    chunk_size: int,
    overlap: int,
) -> tuple[list[Chunk], list[tuple[int, str]]]:
    chunks: list[Chunk] = []
    current_section_path = list(initial_section_path)

    def formatted_section() -> str:
        return " > ".join(title for _, title in current_section_path) or "未识别章节"

    segment_section = formatted_section()
    segment_lines: list[str] = []
    sequence = 1

    def flush_segment() -> None:
        nonlocal sequence, segment_lines
        for text in _split_lines_with_overlap(segment_lines, chunk_size, overlap):
            chunks.append(
                Chunk(
                    chunk_id=f"{document_id}-p{page_number:04d}-c{sequence:03d}",
                    document_id=document_id,
                    source_name=source_name,
                    page=page_number,
                    section=segment_section or "未识别章节",
                    text=text,
                )
            )
            sequence += 1
        segment_lines = []

    for line in lines:
        heading = detect_section(line)
        if heading:
            level = section_level(line)
            assert level is not None
            segment_length = sum(len(item) + 1 for item in segment_lines)

            should_flush = bool(segment_lines) and (
                level <= 2 or segment_length >= max(240, chunk_size // 3)
            )
            if should_flush:
                flush_segment()

            while current_section_path and current_section_path[-1][0] >= level:
                current_section_path.pop()
            current_section_path.append((level, heading))
            segment_section = formatted_section()
        segment_lines.append(line)

    if segment_lines:
        flush_segment()
    return chunks, current_section_path


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def extract_pdf(
    pdf_path: Path,
    data_dir: Path,
    *,
    chunk_size: int = 800,
    overlap: int = 120,
    margin_lines: int = 2,
) -> dict[str, Any]:
    pdf_path = pdf_path.resolve()
    document_id = make_document_id(pdf_path)
    raw_pages: list[list[str]] = []
    embedded_empty_pages: list[int] = []
    ocr_page_indices: list[int] = []
    ocr: PageOCR | None = None

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            deduped_page = page.dedupe_chars(tolerance=1, extra_attrs=("fontname", "size"))
            text = deduped_page.extract_text(x_tolerance=2, y_tolerance=3) or ""
            lines = text.splitlines()
            # 无内嵌文字层的页面尝试 OCR 补全（扫描件）；OCR 依赖缺失时保持空页
            if not any(line.strip() for line in lines):
                embedded_empty_pages.append(page_number)
                if ocr is None:
                    ocr = PageOCR(pdf_path)
                if ocr.available:
                    ocr_text = ocr.ocr(page_number - 1)
                    if ocr_text.strip():
                        lines = ocr_text.splitlines()
                        ocr_page_indices.append(page_number)
            raw_pages.append(lines)

    if ocr is not None:
        ocr.close()

    repeated_margin_keys = find_repeated_margin_keys(raw_pages, margin_lines=margin_lines)
    page_count = len(raw_pages)
    extractable_text_pages = sum(
        1 for raw_lines in raw_pages if "\n".join(raw_lines).strip()
    )
    empty_text_pages = [
        page_number
        for page_number, raw_lines in enumerate(raw_pages, start=1)
        if not "\n".join(raw_lines).strip()
    ]
    embedded_text_page_count = page_count - len(embedded_empty_pages)
    embedded_text_layer_coverage = (
        embedded_text_page_count / page_count if page_count else 0.0
    )
    # 最终可检索覆盖率 = 内嵌文字层 + OCR 补全
    text_layer_coverage = (
        extractable_text_pages / page_count if page_count else 0.0
    )
    page_records: list[dict[str, Any]] = []
    chunk_records: list[dict[str, Any]] = []
    current_section_path: list[tuple[int, str]] = []

    for page_number, raw_lines in enumerate(raw_pages, start=1):
        lines = clean_page_lines(raw_lines, repeated_margin_keys)
        page_text = "\n".join(lines)
        page_records.append(
            {
                "document_id": document_id,
                "source_name": pdf_path.name,
                "page": page_number,
                "text": page_text,
            }
        )

        page_chunks, current_section_path = _chunk_page(
            lines=lines,
            document_id=document_id,
            source_name=pdf_path.name,
            page_number=page_number,
            initial_section_path=current_section_path,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        chunk_records.extend(chunk.to_dict() for chunk in page_chunks if chunk.text.strip())

    pages_path = data_dir / "pages" / f"{document_id}.jsonl"
    chunks_path = data_dir / "chunks" / f"{document_id}.jsonl"
    _write_jsonl(pages_path, page_records)
    _write_jsonl(chunks_path, chunk_records)

    stat = pdf_path.stat()
    return {
        "document_id": document_id,
        "source_name": pdf_path.name,
        "source_path": str(pdf_path),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "page_count": len(page_records),
        "chunk_count": len(chunk_records),
        "text_char_count": sum(len(page["text"]) for page in page_records),
        "extractable_text_page_count": extractable_text_pages,
        "empty_text_page_count": len(empty_text_pages),
        "empty_text_pages": empty_text_pages,
        "text_layer_coverage": text_layer_coverage,
        "embedded_text_layer_coverage": embedded_text_layer_coverage,
        "ocr_page_count": len(ocr_page_indices),
        "ocr_page_indices": ocr_page_indices,
        "chunks_path": str(chunks_path.resolve()),
        "pages_path": str(pages_path.resolve()),
        "extractor": "pdfplumber",
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "extractor_schema_version": EXTRACTOR_SCHEMA_VERSION,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "margin_lines": margin_lines,
        },
    }


def find_pdfs(pdf_dir: Path) -> list[Path]:
    unique: dict[str, Path] = {}
    for path in pdf_dir.iterdir():
        if path.is_file() and path.suffix.lower() == ".pdf":
            unique[str(path.resolve()).casefold()] = path
    return sorted(unique.values(), key=lambda item: item.name.casefold())


def build_corpus(
    pdf_dir: Path,
    data_dir: Path,
    *,
    chunk_size: int = 800,
    overlap: int = 120,
    margin_lines: int = 2,
    force: bool = False,
) -> list[dict[str, Any]]:
    if overlap < 0 or chunk_size <= 0 or overlap >= chunk_size:
        raise ValueError("必须满足 0 <= overlap < chunk_size")

    manifest_path = data_dir / "manifest.json"
    existing: dict[str, dict[str, Any]] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing = {doc["source_path"]: doc for doc in manifest.get("documents", [])}

    documents: list[dict[str, Any]] = []
    for pdf_path in find_pdfs(pdf_dir):
        resolved = str(pdf_path.resolve())
        stat = pdf_path.stat()
        cached = existing.get(resolved)

        settings_match = cached and cached.get("settings") == {
            "extractor_schema_version": EXTRACTOR_SCHEMA_VERSION,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "margin_lines": margin_lines,
        }
        source_match = (
            cached
            and cached.get("source_size") == stat.st_size
            and cached.get("source_mtime_ns") == stat.st_mtime_ns
        )
        output_exists = cached and Path(cached["chunks_path"]).exists()

        if not force and settings_match and source_match and output_exists:
            print(f"[跳过] 索引未变化：{pdf_path.name}")
            documents.append(cached)
            continue

        print(f"[提取] {pdf_path.name}")
        document = extract_pdf(
            pdf_path,
            data_dir,
            chunk_size=chunk_size,
            overlap=overlap,
            margin_lines=margin_lines,
        )
        print(
            f"[完成] {document['page_count']} 页，"
            f"{document['chunk_count']} 块，{document['text_char_count']} 字符，"
            f"文字层覆盖 {document['extractable_text_page_count']}/"
            f"{document['page_count']} 页 "
            f"({document['text_layer_coverage']:.1%})"
            f"{'，OCR 补全 ' + str(document['ocr_page_count']) + ' 页' if document['ocr_page_count'] else ''}"
        )
        if document["embedded_text_layer_coverage"] < 0.9:
            if document["ocr_page_count"]:
                print(
                    f"[OCR] {pdf_path.name} 有 {document['ocr_page_count']} 页无内嵌文字层，"
                    "已用 OCR 补全，重要数字建议对照原文复核。"
                )
            else:
                print(
                    f"[警告] {pdf_path.name} 有 {document['empty_text_page_count']} 页无文字层"
                    "且 OCR 不可用，可能是扫描版 PDF。请安装 rapidocr-onnxruntime 后重新 build。"
                )
        documents.append(document)

    data_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "documents": documents,
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, manifest_path)
    return documents
