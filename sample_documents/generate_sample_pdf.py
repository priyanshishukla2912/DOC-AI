"""
Generator for DocuChat Sample Test PDF
Creates a rich 4-page test PDF document with realistic content for RAG testing.
"""

import os
import pypdf

def generate_test_pdf(output_path: str):
    objects = []

    def add_obj(content):
        objects.append(content)
        return len(objects)

    # Object 1: Font
    font_id = add_obj("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    # Reserve Object 2 for Pages catalog
    pages_id = 2
    objects.append(None)
    catalog_id = add_obj(f"<< /Type /Catalog /Pages {pages_id} 0 R >>")

    pages_data = [
        # PAGE 1
        [
            "DOCUCHAT PLATFORM VERIFICATION AND BENCHMARK SPECIFICATION",
            "Document ID: DOC-SPEC-2026-V1 | Classification: Confidential Research",
            "Author: Priyanshi Shukla | System: DocuChat AI-Powered Document Q&A",
            "",
            "CHAPTER 1: EXECUTIVE SYSTEM SUMMARY",
            "DocuChat is an enterprise-grade conversational AI platform designed to transform static corporate",
            "documents into interactive, searchable knowledge bases. Unlike standard generative chatbots,",
            "DocuChat guarantees strict factual grounding by enforcing a deterministic Retrieval-Augmented",
            "Generation (RAG) pipeline.",
            "",
            "Key Architectural Pillars:",
            "1. End-to-End Cryptographic Privacy: All user uploads are encrypted at rest with AES-256-CBC.",
            "2. Volatile Memory Decryption: Document buffers are decrypted only within volatile RAM during vector",
            "   indexing and are never exposed as plaintext on physical storage media.",
            "3. Pinpoint Source Attribution: Every model claim links directly to its source document and page number.",
            "4. Anti-Hallucination Guardrails: Prevents speculative responses when documents lack relevant context.",
            "",
            "Expected Benchmark Result for Page 1:",
            "When queried about 'What are the key architectural pillars?', the system should return all four points."
        ],
        # PAGE 2
        [
            "CHAPTER 2: CRYPTOGRAPHIC VAULT AND DATA PRIVACY STANDARDS",
            "",
            "2.1 SYMMETRIC ENCRYPTION AT REST (AES-256)",
            "DocuChat implements the Fernet authenticated symmetric cipher suite, which combines AES-128-CBC",
            "with HMAC-SHA256 for authenticated message integrity. The system derives high-entropy 256-bit keys",
            "using PBKDF2-HMAC-SHA256 with 200,000 hashing iterations and unique cryptographic salts.",
            "",
            "2.2 ZERO-RETENTION MEMORY SCRUBBING",
            "When the user initiates a Zero-Retention Purge:",
            "- All encrypted .enc files stored in the .vault directory undergo multi-pass pseudorandom overwrites.",
            "- In-memory cryptographic keys and vector embeddings are flushed from volatile memory.",
            "- The key fingerprint is regenerated, invalidating prior session context permanently.",
            "",
            "2.3 PII ANONYMIZATION AND REDACTION",
            "To safeguard user privacy, DocuChat features an automated Personally Identifiable Information (PII) scrubber.",
            "Sensitive identifiers including email addresses, telephone numbers, and Social Security Numbers are",
            "masked before text context is submitted to the Large Language Model.",
            "",
            "Expected Benchmark Result for Page 2:",
            "When asked 'How does zero-retention purge work?', the system should cite the multi-pass overwrite on Page 2."
        ],
        # PAGE 3
        [
            "CHAPTER 3: RETRIEVAL-AUGMENTED GENERATION (RAG) PIPELINE",
            "",
            "3.1 RECURSIVE SEMANTIC TEXT CHUNKING",
            "Documents are segmented using a recursive character splitting algorithm with a default chunk size of",
            "700 characters and a 120-character sliding overlap. This ensures semantic continuity across sentence boundaries.",
            "",
            "3.2 DUAL-MODE HYBRID VECTOR INDEX",
            "The vector engine operates in two interchangeable modes:",
            "- Neural Mode: Powered by Google Gemini text-embedding-004 generating 768-dimensional dense vectors.",
            "- In-Memory Sparse Mode: An optimized TF-IDF and BM25 cosine similarity index that functions fully offline",
            "  with zero external API dependency.",
            "",
            "3.3 TOP-K RETRIEVAL AND RELEVANCE SCORING",
            "User queries are mapped into the vector space. The retriever ranks candidates using normalized cosine",
            "similarity and returns the top 4 most relevant chunks with confidence percentages.",
            "",
            "Expected Benchmark Result for Page 3:",
            "When asked 'What chunk size and overlap does the RAG pipeline use?', the system should cite 700 and 120."
        ],
        # PAGE 4
        [
            "CHAPTER 4: PERFORMANCE BENCHMARKS AND VERIFICATION METRICS",
            "",
            "4.1 LATENCY AND THROUGHPUT TARGETS",
            "- Average Ingestion & Encryption Latency: 120 milliseconds per document page.",
            "- Average Vector Retrieval Latency: 45 milliseconds across 5,000 indexed chunks.",
            "- Streaming Response First-Token Latency: 350 milliseconds using Gemini 1.5 Flash.",
            "- Factual Grounding Precision Score: 99.4% on standard legal and technical question benchmarks.",
            "",
            "4.2 PROJECT CREDITS AND ATTRIBUTION",
            "Project Name: DocuChat — AI-Powered Document Q&A System",
            "Lead Architect and Engineer: Priyanshi Shukla",
            "Repository: https://github.com/priyanshishukla2912/DOC-AI",
            "Official Signature: MADE WITH HEART EMOJI PRIYANSHI SHUKLA",
            "",
            "End of Specification Document - Ready for Ingestion Testing."
        ]
    ]

    page_ids = []
    content_ids = []

    for page_lines in pages_data:
        stream_cmds = ["BT", "/F1 11 Tf", "16 TL", "50 740 Td"]
        for line in page_lines:
            line_clean = line.replace("—", "--").replace("–", "-").replace("'", "'").replace('"', '"')
            escaped = line_clean.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            # Headers bold-like spacing
            if line_clean.startswith("CHAPTER") or line_clean.startswith("DOCUCHAT"):
                stream_cmds.append("1.5 Tc")
            else:
                stream_cmds.append("0 Tc")
            stream_cmds.append(f"({escaped}) Tj")
            stream_cmds.append("T*")
        stream_cmds.append("ET")

        stream_data = "\n".join(stream_cmds).encode("latin1", errors="replace")
        cid = add_obj(f"<< /Length {len(stream_data)} >>\nstream\n".encode("latin1") + stream_data + b"\nendstream")
        content_ids.append(cid)

        pid = add_obj(
            f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {cid} 0 R >>"
        )
        page_ids.append(pid)

    kids_str = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects[pages_id - 1] = f"<< /Type /Pages /Kids [{kids_str}] /Count {len(page_ids)} >>"

    offsets = []
    with open(output_path, "wb") as f:
        f.write(b"%PDF-1.4\n")
        for i, obj in enumerate(objects):
            offsets.append(f.tell())
            header = f"{i + 1} 0 obj\n".encode("latin1")
            footer = b"\nendobj\n"
            if isinstance(obj, str):
                f.write(header + obj.encode("latin1") + footer)
            else:
                f.write(header + obj + footer)

        startxref = f.tell()
        f.write(b"xref\n")
        f.write(f"0 {len(objects) + 1}\n".encode("latin1"))
        f.write(b"0000000000 65535 f \n")
        for off in offsets:
            f.write(f"{off:010d} 00000 n \n".encode("latin1"))
        f.write(b"trailer\n")
        f.write(f"<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n".encode("latin1"))
        f.write(f"startxref\n{startxref}\n%%EOF\n".encode("latin1"))

    # Verify with pypdf
    reader = pypdf.PdfReader(output_path)
    print(f"Successfully generated {output_path} with {len(reader.pages)} pages.")

if __name__ == "__main__":
    out = "/Users/shukla/.gemini/antigravity/scratch/DOC-AI/sample_documents/DocuChat_Sample_Test_Document.pdf"
    generate_test_pdf(out)
