"""
DocuChat - RAG (Retrieval-Augmented Generation) Pipeline
Author: Priyanshi Shukla
Description: Semantic chunking, dual-mode vector embeddings (Google Gemini & Local TF-IDF/Cosine),
hybrid ranking, and citation-aware context assembly.
"""

import math
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


@dataclass
class TextChunk:
    """Represents an atomic, searchable chunk with full citation lineage."""
    chunk_id: str
    doc_name: str
    page_num: int
    chunk_index: int
    text: str
    char_count: int
    word_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Retrieved document chunk accompanied by semantic relevance scoring."""
    chunk: TextChunk
    score: float  # Normalized 0.0 - 1.0
    rank: int

    @property
    def percentage(self) -> int:
        return min(100, max(0, int(round(self.score * 100))))

    @property
    def relevance_badge(self) -> str:
        if self.score >= 0.70:
            return "🟢 High Relevance"
        elif self.score >= 0.40:
            return "🟡 Moderate Relevance"
        else:
            return "⚪ Supplementary Match"


class RecursiveTextSplitter:
    """
    Splits text recursively using semantic boundaries (paragraphs -> sentences -> words)
    to retain coherent context while enforcing chunk size and overlap constraints.
    """

    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 120,
                 separators: Optional[List[str]] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]

    def split_text(self, text: str) -> List[str]:
        """Splits a single text string into overlapping chunks."""
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        return self._recursive_split(text, self.separators)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        final_chunks: List[str] = []
        separator = separators[-1]
        new_separators = []

        for i, sep in enumerate(separators):
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator)
        good_splits: List[str] = []

        for s in splits:
            s_clean = s.strip()
            if not s_clean:
                continue
            if len(s_clean) < self.chunk_size:
                good_splits.append(s_clean)
            else:
                if new_separators:
                    sub_splits = self._recursive_split(s_clean, new_separators)
                    good_splits.extend(sub_splits)
                else:
                    # Hard split if no more separators
                    for start in range(0, len(s_clean), self.chunk_size - self.chunk_overlap):
                        good_splits.append(s_clean[start:start + self.chunk_size])

        # Merge splits up to chunk_size with overlap
        current_chunk: List[str] = []
        current_len = 0

        for split in good_splits:
            candidate_len = current_len + len(split) + (len(separator) if current_chunk else 0)
            if candidate_len <= self.chunk_size:
                current_chunk.append(split)
                current_len = candidate_len
            else:
                if current_chunk:
                    chunk_str = separator.join(current_chunk)
                    final_chunks.append(chunk_str)

                    # Build overlap from the end of current_chunk
                    overlap_chunk: List[str] = []
                    overlap_len = 0
                    for prev in reversed(current_chunk):
                        if overlap_len + len(prev) <= self.chunk_overlap:
                            overlap_chunk.insert(0, prev)
                            overlap_len += len(prev)
                        else:
                            break
                    current_chunk = overlap_chunk
                    current_len = sum(len(x) for x in current_chunk)

                current_chunk.append(split)
                current_len += len(split)

        if current_chunk:
            final_chunks.append(separator.join(current_chunk))

        return [c for c in final_chunks if c.strip()]


class HybridVectorStore:
    """
    High-performance vector index supporting both Gemini neural embeddings
    and an in-memory TF-IDF + BM25 Cosine ranking engine.
    Ensures the RAG pipeline is fully operational both online and offline.
    """

    def __init__(self, gemini_api_key: Optional[str] = None):
        self.gemini_api_key = gemini_api_key
        self.chunks: List[TextChunk] = []
        self.dense_embeddings: Optional[np.ndarray] = None
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.tfidf_matrix: Optional[np.ndarray] = None
        self._is_gemini_active = False

    def clear(self):
        """Wipes all indexed chunks and vector matrices from memory."""
        self.chunks.clear()
        self.dense_embeddings = None
        self.vocab.clear()
        self.idf.clear()
        self.tfidf_matrix = None
        self._is_gemini_active = False

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple lowercase word tokenizer."""
        return re.findall(r'\b[a-zA-Z0-9_-]{2,}\b', text.lower())

    def add_chunks(self, new_chunks: List[TextChunk]):
        """Indexes new document chunks and rebuilds vector indexes."""
        self.chunks.extend(new_chunks)
        self._build_sparse_index()

        # Attempt Gemini embedding if API key is present
        if self.gemini_api_key:
            try:
                self._build_dense_embeddings()
                self._is_gemini_active = True
            except Exception:
                self._is_gemini_active = False

    def _build_sparse_index(self):
        """Computes sublinear TF-IDF vector matrix over indexed chunks."""
        if not self.chunks:
            return

        doc_count = len(self.chunks)
        df: Dict[str, int] = {}
        tokenized_docs = []

        for chunk in self.chunks:
            tokens = set(self._tokenize(chunk.text))
            tokenized_docs.append(self._tokenize(chunk.text))
            for tok in tokens:
                df[tok] = df.get(tok, 0) + 1

        # Keep vocabulary
        self.vocab = {term: idx for idx, (term, freq) in enumerate(df.items()) if freq >= 1}
        vocab_size = len(self.vocab)

        # Compute IDF
        self.idf = {
            term: math.log((doc_count + 1.0) / (df[term] + 1.0)) + 1.0
            for term in self.vocab
        }

        # Build TF-IDF matrix
        matrix = np.zeros((doc_count, vocab_size), dtype=np.float32)
        for i, tokens in enumerate(tokenized_docs):
            tf_counts: Dict[str, int] = {}
            for t in tokens:
                if t in self.vocab:
                    tf_counts[t] = tf_counts.get(t, 0) + 1

            for term, count in tf_counts.items():
                col = self.vocab[term]
                # Sublinear TF scaling (1 + log(tf))
                tf_val = 1.0 + math.log(count)
                matrix[i, col] = tf_val * self.idf[term]

        # L2 Normalize rows
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.tfidf_matrix = matrix / norms

    def _build_dense_embeddings(self):
        """Generates dense vector representations using Google Gemini."""
        import google.generativeai as genai
        genai.configure(api_key=self.gemini_api_key)

        texts = [c.text for c in self.chunks]
        embeddings = []
        batch_size = 20

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = genai.embed_content(
                model="models/text-embedding-004",
                content=batch,
                task_type="retrieval_document"
            )
            embeddings.extend(response['embedding'])

        dense = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.dense_embeddings = dense / norms

    def query(self, query_text: str, top_k: int = 4, doc_filter: Optional[str] = None) -> List[RetrievalResult]:
        """
        Retrieves top_k most relevant chunks matching the query.
        Uses hybrid dense/sparse scoring when available.
        """
        if not self.chunks or self.tfidf_matrix is None:
            return []

        # Filter indices by doc_name if requested
        candidate_indices = [
            i for i, c in enumerate(self.chunks)
            if doc_filter is None or c.doc_name == doc_filter
        ]
        if not candidate_indices:
            return []

        # Compute query sparse TF-IDF vector
        query_tokens = self._tokenize(query_text)
        q_vec = np.zeros((1, len(self.vocab)), dtype=np.float32)
        q_counts: Dict[str, int] = {}
        for t in query_tokens:
            if t in self.vocab:
                q_counts[t] = q_counts.get(t, 0) + 1

        for term, count in q_counts.items():
            col = self.vocab[term]
            q_vec[0, col] = (1.0 + math.log(count)) * self.idf[term]

        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec /= q_norm

        # Cosine similarity for sparse
        sparse_scores = np.dot(self.tfidf_matrix[candidate_indices], q_vec.T).flatten()

        # Check if dense Gemini scoring is possible
        if self._is_gemini_active and self.dense_embeddings is not None and self.gemini_api_key:
            try:
                import google.generativeai as genai
                q_emb = genai.embed_content(
                    model="models/text-embedding-004",
                    content=query_text,
                    task_type="retrieval_query"
                )['embedding']
                q_dense = np.array(q_emb, dtype=np.float32).reshape(1, -1)
                q_d_norm = np.linalg.norm(q_dense)
                if q_d_norm > 0:
                    q_dense /= q_d_norm

                dense_scores = np.dot(self.dense_embeddings[candidate_indices], q_dense.T).flatten()
                # Hybrid reciprocal score: 70% dense + 30% sparse
                final_scores = 0.70 * dense_scores + 0.30 * sparse_scores
            except Exception:
                final_scores = sparse_scores
        else:
            final_scores = sparse_scores

        # Rank candidates
        top_k_actual = min(top_k, len(candidate_indices))
        ranked_local_indices = np.argsort(-final_scores)[:top_k_actual]

        results: List[RetrievalResult] = []
        for rank, local_idx in enumerate(ranked_local_indices):
            actual_chunk_idx = candidate_indices[local_idx]
            raw_score = float(final_scores[local_idx])
            # Bound score between 0.0 and 1.0
            norm_score = max(0.0, min(1.0, raw_score))

            results.append(
                RetrievalResult(
                    chunk=self.chunks[actual_chunk_idx],
                    score=norm_score,
                    rank=rank + 1
                )
            )

        return results

    @staticmethod
    def build_grounded_context(results: List[RetrievalResult]) -> str:
        """
        Formats retrieved snippets into an indexed prompt block with explicit
        document and page citations.
        """
        if not results:
            return "No matching document context was retrieved."

        blocks = []
        for r in results:
            c = r.chunk
            header = f"[Source Citation: {c.doc_name} | Page/Section: {c.page_num} | Relevance: {r.percentage}%]"
            blocks.append(f"{header}\n{c.text}")

        return "\n\n---\n\n".join(blocks)
