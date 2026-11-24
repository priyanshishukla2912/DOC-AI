"""
Unit tests for RAG Pipeline: Recursive Text Splitter, Hybrid Vector Store, and Retriever
"""

import unittest
from modules.rag_pipeline import RecursiveTextSplitter, HybridVectorStore, TextChunk


class TestRAGPipeline(unittest.TestCase):
    def test_recursive_text_splitter(self):
        long_text = (
            "Paragraph 1 discusses cybersecurity and cryptographic protocols in modern banking.\n\n"
            "Paragraph 2 explores distributed ledger technology, consensus mechanisms, and fault tolerance.\n\n"
            "Paragraph 3 covers large language model alignment, direct preference optimization, and safety."
        )
        splitter = RecursiveTextSplitter(chunk_size=120, chunk_overlap=30)
        chunks = splitter.split_text(long_text)

        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertTrue(len(c) <= 150)  # within reasonable bound

    def test_vector_store_retrieval(self):
        store = HybridVectorStore()
        chunks = [
            TextChunk(
                chunk_id="chunk-1",
                doc_name="finance.txt",
                page_num=1,
                chunk_index=1,
                text="The company's quarterly revenue surged by 35 percent due to cloud adoption.",
                char_count=75,
                word_count=12
            ),
            TextChunk(
                chunk_id="chunk-2",
                doc_name="robotics.txt",
                page_num=1,
                chunk_index=1,
                text="Bipedal humanoid robots utilize inverse kinematics and harmonic drive actuators.",
                char_count=80,
                word_count=10
            )
        ]
        store.add_chunks(chunks)

        # Query relevant to finance
        results = store.query("What was the quarterly revenue growth?", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk.chunk_id, "chunk-1")
        self.assertGreater(results[0].score, 0.2)

        # Build grounded context
        context = HybridVectorStore.build_grounded_context(results)
        self.assertIn("finance.txt", context)
        self.assertIn("Page/Section: 1", context)


if __name__ == "__main__":
    unittest.main()
