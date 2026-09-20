from __future__ import annotations

from pathlib import Path


def _dependencies() -> tuple | None:
    """延迟导入 OCR 依赖，未安装时不影响纯文字层流程。"""
    try:
        import fitz
        from rapidocr_onnxruntime import RapidOCR
    except Exception:
        return None
    return fitz, RapidOCR


def ocr_dependencies_available() -> bool:
    return _dependencies() is not None


class PageOCR:
    """把 PDF 页面渲染成图片并用 RapidOCR 识别，用于补全无文字层的扫描页。"""

    def __init__(self, pdf_path: Path) -> None:
        deps = _dependencies()
        self._fitz = None
        self._engine = None
        self._doc = None
        self.available = deps is not None
        if deps is None:
            return
        fitz, RapidOCR = deps
        try:
            self._fitz = fitz
            self._engine = RapidOCR()
            self._doc = fitz.open(str(pdf_path))
        except Exception:
            self.available = False
            self.close()

    def ocr(self, page_index: int) -> str:
        if not self.available or self._doc is None or self._engine is None:
            return ""
        if page_index < 0 or page_index >= self._doc.page_count:
            return ""
        page = self._doc.load_page(page_index)
        # 2 倍缩放 ≈ 144 DPI，兼顾识别率与耗时
        pix = page.get_pixmap(matrix=self._fitz.Matrix(2.0, 2.0))
        result, _elapse = self._engine(pix.tobytes("png"))
        if not result:
            return ""
        return "\n".join(str(item[1]) for item in result)

    def close(self) -> None:
        if self._doc is not None:
            try:
                self._doc.close()
            except Exception:
                pass
        self._doc = None
