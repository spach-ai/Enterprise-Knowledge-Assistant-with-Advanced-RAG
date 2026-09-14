import os
from typing import Any, Dict, List,Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from .config import TOP_K_FINAL

def create_llm(provider: Optional[str] = None):
    """
    provider can be: openai or gemini
    If provider is not provided, LLM_PROVIDER is used.
    """

    provider = (
        provider or os.getenv("LLM_PROVIDER", "openai")
    ).strip().lower()

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.1,
            api_key=api_key,
        )

    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")

        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            temperature=0.1,
            google_api_key=api_key,
        )

    raise ValueError(
        f"Unsupported provider: {provider}. "
        "Choose either 'openai' or 'gemini'."
    )

def build_context_block(chunks: List[Dict[str, Any]]) -> str:
    parts = []
    for i, chunk in enumerate(chunks[:TOP_K_FINAL], start=1):
        meta = chunk["metadata"]
        source_name = meta.get("file_name", "unknown")
        page = meta.get("page", None)

        if isinstance(page, int):
            source_text = f"{source_name} (p. {page + 1})"
        else:
            source_text = source_name

        parts.append(
            f"[Context {i}] Source: {source_text}\n"
            f"{chunk['content']}"
        )

    return "\n\n".join(parts)


def generate_answer(llm, question: str, context_chunks: List[Dict[str, Any]], history_text: str = "") -> str:
    if not context_chunks:
        return (
        "I couldn't find relevant information in the provided documents. "
        "Please ask about company policies, HR, IT, leave, travel, benefits, or related topics."
    )

    context_block = build_context_block(context_chunks)

    system_prompt = (
    "You are an enterprise knowledge assistant for internal company documents.\n"
    "Use ONLY the provided context to answer.\n"
    "If the user is asking about company policies or documents and the answer is not supported, say:\n"
    "\"I couldn't find relevant information in the provided documents.\"\n"
    "Do not invent facts or use outside knowledge.\n"
    "If the question is a greeting or small talk, respond politely and naturally.\n"
    "Be concise, clear, and friendly."
    )

    user_prompt = f'''
Conversation history:
{history_text if history_text.strip() else "None"}

User question:
{question}

Retrieved context:
{context_block}

Answer the question based only on the retrieved context.
'''.strip()

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    return response.content.strip()
