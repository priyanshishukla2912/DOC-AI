"""
DocuChat - Gemini LLM & Conversational RAG Client
Author: Priyanshi Shukla
Description: Connects to Google Gemini (1.5 Flash/Pro, 2.0) with anti-hallucination prompt
templates, streaming generation, conversational memory, executive summaries, and smart question generation.
"""

import os
from typing import List, Dict, Any, Generator, Optional
import google.generativeai as genai


SYSTEM_INSTRUCTION = """You are DocuChat, an advanced, privacy-first Document Intelligence and Q&A Assistant built by Priyanshi Shukla.

Your primary directive is to answer user queries truthfully, concisely, and comprehensively based SOLELY on the provided retrieved document context.

Strict Guidelines:
1. Grounding: Rely strictly on the context provided under '--- RETRIEVED DOCUMENT CONTEXT ---'. Do not extrapolate, assume, or hallucinate facts not present in the text.
2. Citation: Explicitly reference the source document name and page/section number whenever providing a claim or fact (e.g., '[Annual_Report.pdf, Page 4]').
3. Missing Information: If the provided context does not contain sufficient information to answer the question, state: "Based on the uploaded documents, I could not find information regarding [topic]."
4. Formatting: Use clean markdown with clear headers, bullet points, and bold text for readability.
5. Tone: Professional, analytical, helpful, and respectful.
"""


class GeminiRAGClient:
    """
    Client interface for Google Gemini models integrated with DocuChat RAG pipeline.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-1.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model_name = model_name
        self._is_configured = False

        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self._is_configured = True
            except Exception:
                self._is_configured = False

    def is_ready(self) -> bool:
        return self._is_configured and bool(self.api_key)

    def generate_rag_stream(
        self,
        query: str,
        context: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.2
    ) -> Generator[str, None, None]:
        """
        Streams generated responses token by token.
        Falls back to local synthesis engine if API key is not configured.
        """
        if not self.is_ready():
            for token in self._generate_fallback_stream(query, context):
                yield token
            return

        # Format prompt with conversational history
        history_text = ""
        if chat_history:
            recent_turns = chat_history[-6:]  # Last 3 turns (user + assistant)
            formatted = []
            for msg in recent_turns:
                role = "User" if msg["role"] == "user" else "Assistant"
                formatted.append(f"{role}: {msg['content']}")
            history_text = "--- RECENT CONVERSATION HISTORY ---\n" + "\n".join(formatted) + "\n\n"

        full_prompt = f"""{history_text}--- RETRIEVED DOCUMENT CONTEXT ---
{context}

--- USER QUESTION ---
{query}

Please provide an accurate, grounded answer with source citations based on the context above:"""

        try:
            model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=SYSTEM_INSTRUCTION,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=2048,
                )
            )
            response = model.generate_content(full_prompt, stream=True)
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            yield f"\n\n⚠️ **Gemini API Error:** {str(e)}\n\n*Falling back to offline extractive synthesis:*\n\n"
            for token in self._generate_fallback_stream(query, context):
                yield token

    def _generate_fallback_stream(self, query: str, context: str) -> Generator[str, None, None]:
        """
        Deterministic, offline fallback synthesis when no Gemini API key is configured.
        Extracts key sentences matching the query terms.
        """
        lines = [
            "**[Offline / Local Mode Active]** *(To enable Google Gemini neural generation, please provide a Gemini API Key in the sidebar.)*\n\n",
            "Here is the synthesized excerpt from your retrieved document context:\n\n"
        ]
        for line in lines:
            yield line

        if not context or context == "No matching document context was retrieved.":
            yield "No relevant information found matching your query in the uploaded documents."
            return

        # Present the retrieved context in clean bullet points
        snippets = context.split("\n\n---\n\n")
        for i, snippet in enumerate(snippets[:3]):
            yield f"### Match {i + 1}\n{snippet}\n\n"

    def generate_executive_summary(self, sample_text: str) -> Dict[str, Any]:
        """
        Generates an executive brief, core topics, and target audience for uploaded documents.
        """
        if not self.is_ready():
            # Local summary heuristic
            words = sample_text.split()
            first_chunk = " ".join(words[:120]) + "..." if len(words) > 120 else sample_text
            return {
                "summary": f"This document covers key concepts beginning with: '{first_chunk}'. (Add your Gemini API Key in the sidebar for full AI executive summaries).",
                "topics": ["Document Analysis", "Data Overview", "Text Structure"],
                "audience": "General / Research Audience"
            }

        prompt = f"""Analyze the following excerpt from an uploaded document and generate a structured JSON-like summary:
Excerpt:
{sample_text[:4000]}

Provide:
1. Executive Summary: A concise 3-sentence summary of the document's main thesis or purpose.
2. Key Topics: 4 to 6 core themes or keywords separated by commas.
3. Target Audience: The primary intended reader.

Format strictly as:
SUMMARY: <3 sentences>
TOPICS: <comma-separated topics>
AUDIENCE: <target audience>"""

        try:
            model = genai.GenerativeModel(model_name="gemini-1.5-flash")
            response = model.generate_content(prompt)
            text = response.text or ""

            summary = "Summary could not be parsed."
            topics = ["Overview", "Analysis"]
            audience = "General Reader"

            for line in text.split("\n"):
                line = line.strip()
                if line.startswith("SUMMARY:"):
                    summary = line.replace("SUMMARY:", "").strip()
                elif line.startswith("TOPICS:"):
                    raw_topics = line.replace("TOPICS:", "").strip()
                    topics = [t.strip() for t in raw_topics.split(",") if t.strip()]
                elif line.startswith("AUDIENCE:"):
                    audience = line.replace("AUDIENCE:", "").strip()

            return {
                "summary": summary,
                "topics": topics,
                "audience": audience
            }
        except Exception:
            return {
                "summary": "Document ingested and ready for natural-language questioning.",
                "topics": ["Document Search", "RAG Pipeline", "Text Ingestion"],
                "audience": "Knowledge Worker"
            }

    def generate_suggested_questions(self, sample_text: str) -> List[str]:
        """
        Generates 3 to 4 context-relevant questions that users can click to ask immediately.
        """
        if not self.is_ready():
            return [
                "What is the main objective or conclusion of this document?",
                "Can you summarize the key findings or core arguments?",
                "What are the most important recommendations or next steps mentioned?",
                "What methodology or approach is described in the text?"
            ]

        prompt = f"""Based on the following document excerpt, generate 4 insightful, natural-language questions that a user might want to ask about this content.

Excerpt:
{sample_text[:3500]}

Format as a simple numbered list, one question per line. Do not include introductory text."""

        try:
            model = genai.GenerativeModel(model_name="gemini-1.5-flash")
            response = model.generate_content(prompt)
            lines = [l.strip() for l in response.text.split("\n") if l.strip()]
            questions = []
            for l in lines:
                # Strip leading numbers or bullets (e.g., "1. ", "- ")
                clean_q = re.sub(r'^\d+[\.\)]\s*', '', l).lstrip('-*• ').strip()
                if clean_q and clean_q.endswith('?'):
                    questions.append(clean_q)
            return questions[:4] if questions else [
                "What is the core message of this document?",
                "What key recommendations are provided?",
                "What are the main risks or benefits highlighted?",
                "Who are the key stakeholders mentioned?"
            ]
        except Exception:
            return [
                "What is the core message of this document?",
                "What key recommendations are provided?",
                "What are the main risks or benefits highlighted?",
                "Who are the key stakeholders mentioned?"
            ]
