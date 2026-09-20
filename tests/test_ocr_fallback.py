import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from annual_report_rag.extractor import build_corpus, extract_pdf


def pdf_pages(*texts):
    context = MagicMock()
    pages = []
    for text in texts:
        page = MagicMock()
        page.dedupe_chars.return_value.extract_text.return_value = text
        pages.append(page)
    context.__enter__.return_value = SimpleNamespace(pages=pages)
    return context


class OCRFallbackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'report.pdf'
        self.source.write_bytes(b'fixture')
        self.data = self.root / 'data'

    def test_installing_ocr_retries_incomplete_cached_document(self):
        engine = MagicMock(available=False)
        with patch('annual_report_rag.extractor.pdfplumber.open', return_value=pdf_pages('')), \
             patch('annual_report_rag.extractor.PageOCR', return_value=engine), \
             patch('annual_report_rag.extractor.ocr_dependencies_available', return_value=False):
            before = build_corpus(self.root, self.data)[0]
        self.assertEqual(before['chunk_count'], 0)
        engine.available = True
        engine.ocr.return_value = '资本充足率 14.30%'
        with patch('annual_report_rag.extractor.pdfplumber.open', return_value=pdf_pages('')), \
             patch('annual_report_rag.extractor.PageOCR', return_value=engine), \
             patch('annual_report_rag.extractor.ocr_dependencies_available', return_value=True):
            after = build_corpus(self.root, self.data)[0]
        self.assertGreater(after['chunk_count'], 0)
        self.assertEqual(after['ocr_page_indices'], [1])
        with patch('annual_report_rag.extractor.extract_pdf') as extract:
            build_corpus(self.root, self.data)
            extract.assert_not_called()

    def test_page_failure_preserves_other_pages_and_closes_engine(self):
        engine = MagicMock(available=True)
        engine.ocr.side_effect = [RuntimeError('inference failed'), '拨备覆盖率 373.16%']
        with patch('annual_report_rag.extractor.pdfplumber.open', return_value=pdf_pages('', '', '原有文字保持不变')), \
             patch('annual_report_rag.extractor.PageOCR', return_value=engine):
            result = extract_pdf(self.source, self.data)
        self.assertEqual(result['ocr_failed_pages'], [1])
        self.assertEqual(result['empty_text_pages'], [1])
        self.assertEqual(result['ocr_page_indices'], [2])
        self.assertEqual(result['extractable_text_page_count'], 2)
        engine.close.assert_called_once()

    def test_text_only_pdf_does_not_initialize_ocr(self):
        with patch('annual_report_rag.extractor.pdfplumber.open', return_value=pdf_pages('资本充足率 14.30%')), \
             patch('annual_report_rag.extractor.PageOCR') as ocr:
            result = extract_pdf(self.source, self.data)
        ocr.assert_not_called()
        self.assertEqual(result['ocr_page_count'], 0)
        self.assertEqual(result['embedded_text_layer_coverage'], 1.0)


if __name__ == '__main__':
    unittest.main()
