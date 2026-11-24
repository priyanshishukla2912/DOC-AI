"""
DocuChat - Multi-Format Document Parser
Author: Priyanshi Shukla
Description: Parses PDF, DOCX, TXT, and Markdown files from decrypted in-memory byte streams,
preserving page-level metadata for pinpoint RAG source attribution.
"""

import io
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pypdf
import docx


@dataclass
class DocumentSegment:
    """Represents a discrete segment or page from an ingested document."""
    doc_name: str
    segment_type: str  # 'page', 'paragraph', 'section'
    page_num: int      # 1-indexed page or logical section
    text: str
    char_count: int
    word_count: int


@dataclass
class DocumentMetadata:
    """Document-level profile and statistical summary."""
    filename: str
    file_type: str
    total_pages: int
    total_words: int
    total_chars: int
    estimated_read_time_min: float
    raw_size_bytes: int


class DocumentParser:
    """
    Parses documents from in-memory byte streams without saving plaintext to disk.
    Supports PDF, DOCX, TXT, and MD formats with page-level granularity.
    """

    @classmethod
    def parse_stream(cls, filename: str, file_bytes: bytes) -> Tuple[List[DocumentSegment], DocumentMetadata]:
        """
        Routes the file bytes to the appropriate format parser based on extension.
        """
        ext = filename.lower().split(".")[-1]
        segments: List[DocumentSegment] = []

        if ext == "pdf":
            segments = cls._parse_pdf(filename, file_bytes)
        elif ext in ("docx", "doc"):
            segments = cls._parse_docx(filename, file_bytes)
        elif ext in ("txt", "md", "markdown", "csv", "json"):
            segments = cls._parse_text(filename, file_bytes, ext)
        else:
            # Fallback to UTF-8 decoded text
            segments = cls._parse_text(filename, file_bytes, ext)

        # Compute document-level statistics
        total_words = sum(s.word_count for s in segments)
        total_chars = sum(s.char_count for s in segments)
        total_pages = max([s.page_num for s in segments], default=1)
        # Average reading speed ~ 220 words per minute
        read_time = round(total_words / 220.0, 1)

        meta = DocumentMetadata(
            filename=filename,
            file_type=ext.upper(),
            total_pages=total_pages,
            total_words=total_words,
            total_chars=total_chars,
            estimated_read_time_min=read_time,
            raw_size_bytes=len(file_bytes),
        )
        return segments, meta

    @staticmethod
    def _parse_pdf(filename: str, file_bytes: bytes) -> List[DocumentSegment]:
        """Extracts text page-by-page from PDF stream."""
        segments: List[DocumentSegment] = []
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))

        for idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            clean_text = page_text.strip()
            if not clean_text:
                continue

            words = len(clean_text.split())
            segments.append(
                DocumentSegment(
                    doc_name=filename,
                    segment_type="page",
                    page_num=idx + 1,
                    text=clean_text,
                    char_count=len(clean_text),
                    word_count=words,
                )
            )

        if not segments:
            # Empty or scanned PDF fallback
            segments.append(
                DocumentSegment(
                    doc_name=filename,
                    segment_type="page",
                    page_num=1,
                    text="[Notice: No extractable text detected in this PDF. It may contain scanned images or require OCR.]",
                    char_count=85,
                    word_count=13,
                )
            )
        return segments

    @staticmethod
    def _parse_docx(filename: str, file_bytes: bytes) -> List[DocumentSegment]:
        """Extracts paragraphs and tables from DOCX stream, grouped into logical sections."""
        segments: List[DocumentSegment] = []
        doc = docx.Document(io.BytesIO(file_bytes))

        current_section: List[str] = []
        current_page = 1
        word_accumulator = 0

        # Process paragraphs
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue

            current_section.append(text)
            word_accumulator += len(text.split())

            # Chunk into ~350 word logical pages for citation fidelity
            if word_accumulator >= 350:
                combined = "\n\n".join(current_section)
                segments.append(
                    DocumentSegment(
                        doc_name=filename,
                        segment_type="section",
                        page_num=current_page,
                        text=combined,
                        char_count=len(combined),
                        word_count=word_accumulator,
                    )
                )
                current_section = []
                word_accumulator = 0
                current_page += 1

        # Process tables if any
        table_texts = []
        for table in doc.tables:
            for row in table.rows:
                row_data = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_data:
                    table_texts.append(" | ".join(row_data))

        if table_texts:
            current_section.append("--- Extracted Tables ---\n" + "\n".join(table_texts))
            word_accumulator += sum(len(t.split()) for t in table_texts)

        if current_section:
            combined = "\n\n".join(current_section)
            segments.append(
                DocumentSegment(
                    doc_name=filename,
                    segment_type="section",
                    page_num=current_page,
                    text=combined,
                    char_count=len(combined),
                    word_count=len(combined.split()),
                )
            )

        if not segments:
            segments.append(
                DocumentSegment(
                    doc_name=filename,
                    segment_type="section",
                    page_num=1,
                    text="[Notice: DOCX file contained no extractable textual content.]",
                    char_count=60,
                    word_count=7,
                )
            )
        return segments

    @staticmethod
    def _parse_text(filename: str, file_bytes: bytes, ext: str) -> List[DocumentSegment]:
        """Extracts text from plain text or markdown files."""
        text = file_bytes.decode("utf-8", errors="replace").strip()
        if not text:
            return [
                DocumentSegment(
                    doc_name=filename,
                    segment_type="section",
                    page_num=1,
                    text="[Empty document]",
                    char_count=16,
                    word_count=2,
                )
            ]

        # Break into logical ~400 word segments for citation reference
        paragraphs = text.split("\n\n")
        segments: List[DocumentSegment] = []
        page_num = 1
        current_batch: List[str] = []
        words_in_batch = 0

        for p in paragraphs:
            clean_p = p.strip()
            if not clean_p:
                continue
            p_words = len(clean_p.split())
            current_batch.append(clean_p)
            words_in_batch += p_words

            if words_in_batch >= 400:
                combined = "\n\n".join(current_batch)
                segments.append(
                    DocumentSegment(
                        doc_name=filename,
                        segment_type="section",
                        page_num=page_num,
                        text=combined,
                        char_count=len(combined),
                        word_count=words_in_batch,
                    )
                )
                current_batch = []
                words_in_batch = 0
                page_num += 1

        if current_batch:
            combined = "\n\n".join(current_batch)
            segments.append(
                DocumentSegment(
                    doc_name=filename,
                    segment_type="section",
                    page_num=page_num,
                    text=combined,
                    char_count=len(combined),
                    word_count=len(combined.split()),
                )
            )

        return segments
