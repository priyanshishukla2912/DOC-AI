"""
Unit tests for Document Parsers (PDF, DOCX, TXT)
"""

import unittest
from modules.document_parser import DocumentParser


class TestDocumentParser(unittest.TestCase):
    def test_parse_text_stream(self):
        text_content = (
            "Introduction to Neural Networks.\n\n"
            "Deep learning has transformed computer vision and NLP.\n\n"
            "Convolutional layers capture spatial hierarchies."
        )
        file_bytes = text_content.encode("utf-8")
        segments, meta = DocumentParser.parse_stream("neural_nets.txt", file_bytes)

        self.assertGreater(len(segments), 0)
        self.assertEqual(meta.file_type, "TXT")
        self.assertGreater(meta.total_words, 10)
        self.assertEqual(segments[0].doc_name, "neural_nets.txt")

    def test_parse_empty_stream(self):
        segments, meta = DocumentParser.parse_stream("empty.txt", b"")
        self.assertEqual(len(segments), 1)
        self.assertIn("Empty", segments[0].text)


if __name__ == "__main__":
    unittest.main()
