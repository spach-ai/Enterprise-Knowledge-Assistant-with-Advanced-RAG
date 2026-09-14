import re

import streamlit as st
from dotenv import load_dotenv

from src.config import (
    CHROMA_DIR,
    TOP_K_BM25,
    TOP_K_FINAL,
    TOP_K_VECTOR,
    RAW_DOCS_DIR,
)
from src.ingest import (
    ingest_corpus,
    index_exists,
    load_indexed_chunks,
    load_vectorstore,
)
from src.llm import create_llm, generate_answer
from src.memory import (
    load_memory,
    clear_memory,
    update_memory_from_text,
    answer_from_memory,
    format_history,
    rewrite_question,
    get_memory_summary,
)
from src.reranker import DocumentReranker
from src.retriever import build_bm25_index, hybrid_search
from src.utils import ensure_dir, snippet, source_label


# --------------------------------------------------
# Application setup
# --------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="PP Knowledge Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

ensure_dir(RAW_DOCS_DIR)
ensure_dir(CHROMA_DIR)


# --------------------------------------------------
# Professional UI styling
# --------------------------------------------------

st.markdown(
    """
    <style>
    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap'
    );

    html, body, [class*="css"] {
        font-family: "Inter", sans-serif;
    }

    .stApp {
        background: #f5f7fb;
    }

    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e5e7eb;
    }

    .app-header {
        background: linear-gradient(135deg, #12304a, #2878b5);
        padding: 24px 30px;
        border-radius: 18px;
        margin-bottom: 22px;
        display: flex;
        align-items: center;
        gap: 16px;
        box-shadow: 0 8px 25px rgba(18, 48, 74, 0.18);
    }

    .brand-icon {
        width: 54px;
        height: 54px;
        border-radius: 15px;
        background: #ffffff;
        color: #12304a;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 27px;
        font-weight: 700;
    }

    .header-content h1 {
        color: white;
        margin: 0;
        font-size: 28px;
        font-weight: 700;
    }

    .header-content p {
        color: #d9edf9;
        margin: 5px 0 0 0;
        font-size: 14px;
    }

    .status-badge {
        margin-left: auto;
        padding: 9px 15px;
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.16);
        color: white;
        font-size: 13px;
        white-space: nowrap;
    }

    .welcome-card {
        background: #ffffff;
        padding: 28px;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        margin-bottom: 20px;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.04);
    }

    .welcome-card h2 {
        color: #12304a;
        margin: 0 0 8px 0;
        font-size: 23px;
    }

    .welcome-card p {
        color: #64748b;
        margin: 0;
        font-size: 15px;
    }

    .source-card {
        background: #f1f8fc;
        border-left: 4px solid #2878b5;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
    }

    .source-title {
        color: #12304a;
        font-weight: 600;
        font-size: 14px;
    }

    .source-subtitle {
        color: #64748b;
        font-size: 12px;
        margin-top: 3px;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        min-height: 80px;
    }

    .metric-number {
        color: #2878b5;
        font-size: 24px;
        font-weight: 700;
    }

    .metric-label {
        color: #64748b;
        font-size: 12px;
        margin-top: 2px;
    }

    .section-label {
        color: #12304a;
        font-size: 15px;
        font-weight: 700;
        margin: 18px 0 8px 0;
    }

    .info-box {
        background: #eef7f2;
        border-left: 4px solid #2e8b68;
        color: #245b45;
        padding: 12px 15px;
        border-radius: 8px;
        font-size: 13px;
    }

    .warning-box {
        background: #fff8eb;
        border-left: 4px solid #df9135;
        color: #805b20;
        padding: 12px 15px;
        border-radius: 8px;
        font-size: 13px;
    }

    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }

    .stChatInput {
        border-radius: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Session state
# --------------------------------------------------

if "turns" not in st.session_state:
    st.session_state.turns = []

if "memory_enabled" not in st.session_state:
    st.session_state.memory_enabled = True

if "memory_store" not in st.session_state:
    st.session_state.memory_store = load_memory()

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def is_personal_fact_statement(text: str) -> bool:
    """
    Detects pure personal statements.

    True:
        My name is Bob
        My name Bob
        I am Bob
        I'm Bob

    False:
        My name is Bob. What is the leave policy?
    """

    text = text.strip().lower()

    patterns = [
        r"^my name(?: is)? [^.!?]+[.!]?$",
        r"^i am [^.!?]+[.!]?$",
        r"^i'm [^.!?]+[.!]?$",
        r"^my department(?: is)? [^.!?]+[.!]?$",
        r"^my role(?: is)? [^.!?]+[.!]?$",
        r"^i work in [^.!?]+[.!]?$",
        r"^i work at [^.!?]+[.!]?$",
    ]

    return any(
        re.match(pattern, text)
        for pattern in patterns
    )

def is_memory_question(text: str) -> bool:
    question = text.lower().strip()
    question = re.sub(r"\s+", " ", question)

    # Direct personal-information questions
    direct_patterns = [
        r"\bwhat is my name\b",
        r"\bwhat's my name\b",
        r"\bwhats my name\b",
        r"\bcan you tell my name\b",
        r"\bcan you tell me my name\b",
        r"\bcould you tell me my name\b",
        r"\bdo you know my name\b",
        r"\bdo you remember my name\b",
        r"\bcan you remember my name\b",
        r"\bwhat do you know about me\b",
    ]

    if any(
        re.search(pattern, question)
        for pattern in direct_patterns
    ):
        return True

    # Flexible detection for phrases such as:
    # "Can you tell my name without memory?"
    has_my_name = re.search(
        r"\bmy name\b",
        question,
    )

    has_memory_intent = re.search(
        r"\b(can|could|do|does|did|know|remember|tell|recall|without)\b",
        question,
    )

    return bool(has_my_name and has_memory_intent)

def remove_personal_fact_from_question(text: str) -> str:
    """
    Removes a personal introduction before document retrieval.

    Example:
        My name Bob. What is the leave policy?

    Result:
        What is the leave policy?
    """

    patterns = [
        r"^\s*my name(?: is)? [^.!?]+[.!]\s*",
        r"^\s*i am [^.!?]+[.!]\s*",
        r"^\s*i'm [^.!?]+[.!]\s*",
        r"^\s*my department(?: is)? [^.!?]+[.!]\s*",
        r"^\s*my role(?: is)? [^.!?]+[.!]\s*",
        r"^\s*i work in [^.!?]+[.!]\s*",
        r"^\s*i work at [^.!?]+[.!]\s*",
    ]

    cleaned_text = text

    for pattern in patterns:
        cleaned_text = re.sub(
            pattern,
            "",
            cleaned_text,
            flags=re.IGNORECASE,
        )

    return cleaned_text.strip()

def is_greeting_or_smalltalk(text: str) -> bool:
    text = text.lower().strip().rstrip("!?.")

    patterns = [
        r"^hi$",
        r"^hello$",
        r"^hey$",
        r"^good morning$",
        r"^good afternoon$",
        r"^good evening$",
        r"^thanks$",
        r"^thank you$",
        r"^ok$",
        r"^okay$",
    ]

    return any(re.match(pattern, text) for pattern in patterns)


def friendly_smalltalk_response(text: str) -> str:
    text = text.lower().strip().rstrip("?!. ")

    if text in [
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
    ]:
        return (
            "Hello! How can I help you with company policies, HR, IT, "
            "leave, travel, benefits, or FAQs?"
        )

    if text in ["thanks", "thank you"]:
        return (
            "You're welcome! Feel free to ask me about the company documents."
        )

    return (
        "I can help with company documents, including HR, IT, leave, "
        "travel, benefits, and company policies."
    )

def is_assistant_identity_question(text: str) -> bool:
    question = text.lower().strip()
    question = re.sub(r"\s+", " ", question)

    identity_patterns = [
        r"\bwhat\s+is\s+(?:your|you|ur)\s+name\b",
        r"\bwhat(?:'s|s)\s+(?:your|you|ur)\s+name\b",
        r"\bwho\s+are\s+you\b",
        r"\bwhat\s+can\s+you\s+do\b",
    ]

    return any(
        re.search(pattern, question)
        for pattern in identity_patterns
    )

def normalize_personal_fact(text: str) -> str:
    """
    Converts:
        My name Bob

    Into:
        My name is Bob
    """

    return re.sub(
        r"^\s*my name ([A-Za-z][^.!?]*)",
        r"My name is \1",
        text.strip(),
        flags=re.IGNORECASE,
    )

def answer_has_no_relevant_information(answer: str) -> bool:
    answer_lower = answer.lower().strip()

    no_information_phrases = [
        "i couldn't find relevant information",
        "i could not find relevant information",
        "i couldn't find relevant information in the provided documents",
        "i could not find relevant information in the provided documents",
        "not found in the provided documents",
        "not available in the provided context",
        "the provided documents do not contain",
        "the provided context does not contain",
        "i don't have enough information",
        "i do not have enough information",
        "cannot answer based on the provided context",
        "unable to answer based on the provided context",
    ]

    return any(
        phrase in answer_lower
        for phrase in no_information_phrases
    )

@st.cache_resource(show_spinner=False)
def load_rag_components():
    if not index_exists():
        return None

    chunks = load_indexed_chunks()

    if not chunks:
        return None

    vectorstore = load_vectorstore()
    bm25 = build_bm25_index(chunks)
    reranker = DocumentReranker()
    llm = create_llm()

    return {
        "chunks": chunks,
        "vectorstore": vectorstore,
        "bm25": bm25,
        "reranker": reranker,
        "llm": llm,
    }


def reset_conversation():
    st.session_state.turns = []


def reset_saved_memory():
    clear_memory()
    st.session_state.memory_store = load_memory()


def render_source_cards(sources):
    if not sources:
        return

    st.markdown("#### Sources")

    for source in sources:
        st.markdown(
            f"""
            <div class="source-card">
                <div class="source-title">📄 {source}</div>
                <div class="source-subtitle">
                    Retrieved from the PP company knowledge base
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_retrieved_details(retrieved):
    if not retrieved:
        return

    with st.expander("View retrieved context and relevance scores"):
        for index, item in enumerate(retrieved, start=1):
            metadata = item.get("metadata", {})
            source = source_label(metadata)

            st.markdown(f"**{index}. {source}**")
            st.write(snippet(item.get("content", "")))

            score_col1, score_col2, score_col3 = st.columns(3)

            with score_col1:
                st.caption(
                    f"Hybrid score: {item.get('rrf_score', 0.0):.4f}"
                )

            with score_col2:
                if "rerank_score" in item:
                    st.caption(
                        f"Rerank score: {item['rerank_score']:.4f}"
                    )

            with score_col3:
                methods = ", ".join(
                    item.get("retrieval_sources", [])
                )
                st.caption(f"Retrieved by: {methods}")

            st.divider()


# --------------------------------------------------
# Header
# --------------------------------------------------

index_ready = index_exists()

status_text = "● Knowledge Base Ready" if index_ready else "● Index Required"
status_color = "#d9f5e5" if index_ready else "#ffe9bd"

st.markdown(
    f"""
    <div class="app-header">
        <div class="brand-icon">P</div>
        <div class="header-content">
            <h1>PP Knowledge Assistant</h1>
            <p>Search internal policies and company knowledge</p>
        </div>
        <div class="status-badge" style="color:{status_color};">
            {status_text}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.markdown("## Project Controls")

    st.session_state.memory_enabled = st.checkbox(
        "Enable conversational memory",
        value=st.session_state.memory_enabled,
    )

    st.divider()

    st.markdown("### Knowledge Base")

    if index_ready:
        indexed_chunks = load_indexed_chunks()
        indexed_files = sorted(
            {
                item.get("metadata", {}).get(
                    "file_name",
                    "Unknown",
                )
                for item in indexed_chunks
            }
        )

        st.markdown(
            '<div class="info-box">Knowledge base is ready.</div>',
            unsafe_allow_html=True,
        )

        st.write(f"Documents indexed: **{len(indexed_files)}**")
        st.write(f"Chunks indexed: **{len(indexed_chunks)}**")
    else:
        st.markdown(
            """
            <div class="warning-box">
                No index found. Add documents to data/raw_docs and build the index.
            </div>
            """,
            unsafe_allow_html=True,
        )

    rebuild_clicked = st.button(
        "🔄 Build / Refresh Index",
        use_container_width=True,
    )

    st.divider()

    st.markdown("### Conversation")

    clear_clicked = st.button(
        "🧹 Clear Conversation",
        use_container_width=True,
    )

    forget_clicked = st.button(
        "🗑️ Forget Saved Facts",
        use_container_width=True,
    )

    if clear_clicked:
        reset_conversation()
        st.rerun()

    if forget_clicked:
        reset_saved_memory()
        st.success("Saved facts cleared.")
        st.rerun()

    st.divider()

    st.markdown("### Saved Memory")

    with st.expander("View saved facts"):
        memory_summary = get_memory_summary(
            st.session_state.memory_store
        )
        st.text(memory_summary)

    st.divider()

    st.markdown("### Supported Documents")

    st.caption("PDF")
    st.caption("DOCX")
    st.caption("TXT")
    st.caption("Markdown")


# --------------------------------------------------
# Build or refresh index
# --------------------------------------------------

components = None

if rebuild_clicked:
    try:
        st.cache_resource.clear()

        with st.status(
            "Building the company knowledge base...",
            expanded=True,
        ) as status:
            st.write("Loading local documents...")
            stats = ingest_corpus(rebuild=True)
            st.write("Creating document chunks and embeddings...")
            status.update(
                label="Knowledge base created successfully",
                state="complete",
            )

        st.success(
            f"Indexed {stats['documents_loaded']} documents "
            f"into {stats['chunks_created']} chunks."
        )

        st.rerun()

    except Exception as error:
        st.error(f"Index build failed: {error}")

elif index_ready:
    try:
        components = load_rag_components()
    except Exception as error:
        st.error(f"Unable to load RAG components: {error}")


# --------------------------------------------------
# Main welcome section
# --------------------------------------------------

if not st.session_state.turns:
    st.markdown(
        """
        <div class="welcome-card">
            <h2>Welcome to your knowledge assistant</h2>
            <p>
                Ask questions about leave, HR, travel, benefits, IT security,
                remote work, and company policies.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-label">Try asking</div>',
        unsafe_allow_html=True,
    )

    suggestions = [
        "What is the annual leave entitlement?",
        "What is the travel reimbursement limit?",
        "Is multi-factor authentication mandatory?",
        "What benefits are available?",
    ]

    columns = st.columns(4)

    for column, question in zip(columns, suggestions):
        with column:
            if st.button(
                question,
                key=f"suggestion_{question}",
                use_container_width=True,
            ):
                st.session_state.pending_question = question
                st.rerun()


# --------------------------------------------------
# Display previous conversation
# --------------------------------------------------

for turn in st.session_state.turns:
    with st.chat_message("user"):
        st.markdown(turn["user"])

    with st.chat_message("assistant"):
        st.markdown(turn["assistant"])

        render_source_cards(turn.get("sources", []))
        render_retrieved_details(turn.get("retrieved", []))


# --------------------------------------------------
# Get user question
# --------------------------------------------------

typed_question = st.chat_input(
    "Ask about company policies, HR, leave, travel, benefits, or IT..."
)

user_question = typed_question

if st.session_state.pending_question:
    user_question = st.session_state.pending_question
    st.session_state.pending_question = None


# --------------------------------------------------
# Process question
# --------------------------------------------------

if user_question:
    user_question = user_question.strip()

    if not user_question:
        st.warning("Please enter a question.")
        st.stop()

    changed_facts = {}

    # --------------------------------------------------
    # Handle assistant identity questions
    # --------------------------------------------------

    if is_assistant_identity_question(user_question):

        # Save name or other facts if memory is enabled
        if st.session_state.memory_enabled:
            update_memory_from_text(
                user_question,
                st.session_state.memory_store,
            )

        answer = (
            "I am the PP Knowledge Assistant. "
            "I help answer questions using the company knowledge base."
        )

        st.session_state.turns.append(
            {
                "user": user_question,
                "assistant": answer,
                "sources": [],
                "retrieved": [],
                "rewritten_question": user_question,
            }
        )

        st.rerun()

    # --------------------------------------------------
    # Handle simple greetings and small talk
    # --------------------------------------------------

    if is_greeting_or_smalltalk(user_question):
        answer = friendly_smalltalk_response(user_question)

        st.session_state.turns.append(
            {
                "user": user_question,
                "assistant": answer,
                "sources": [],
                "retrieved": [],
                "rewritten_question": user_question,
            }
        )

        st.rerun()

    # --------------------------------------------------
    # Handle memory
    # --------------------------------------------------

    changed_facts = {}

    if st.session_state.memory_enabled:
        memory_input = normalize_personal_fact(user_question)

        changed_facts = update_memory_from_text(
            memory_input,
            st.session_state.memory_store,
        )

        memory_answer = answer_from_memory(
            user_question,
            st.session_state.memory_store,
        )

        if memory_answer and not changed_facts:
            st.session_state.turns.append(
                {
                    "user": user_question,
                    "assistant": memory_answer,
                    "sources": [],
                    "retrieved": [],
                    "rewritten_question": user_question,
                }
            )

            st.rerun()

    # --------------------------------------------------
    # Memory question without an available answer
    # --------------------------------------------------

    if is_memory_question(user_question):

        if st.session_state.memory_enabled:
            memory_answer = answer_from_memory(
                user_question,
                st.session_state.memory_store,
            )

            if memory_answer:
                answer = memory_answer
            else:
                answer = (
                    "I do not have your name saved yet. "
                    "Please tell me your name first."
                )
        else:
            answer = (
                "I cannot know your name when conversational memory is "
                "disabled. Please enable memory and tell me your name first."
            )

        st.session_state.turns.append(
            {
                "user": user_question,
                "assistant": answer,
                "sources": [],
                "retrieved": [],
                "rewritten_question": user_question,
            }
        )

        st.rerun()

    # --------------------------------------------------
    # Handle pure personal statements
    # --------------------------------------------------

    if is_personal_fact_statement(user_question):
        if changed_facts:
            saved_items = ", ".join(
                f"{key}: {value}"
                for key, value in changed_facts.items()
            )

            answer = (
                f"I saved this information: {saved_items}\n\n"
                "How can I assist you today?"
            )
        else:
            answer = (
                "Nice to meet you! How can I assist you today?"
            )

        st.session_state.turns.append(
            {
                "user": user_question,
                "assistant": answer,
                "sources": [],
                "retrieved": [],
                "rewritten_question": user_question,
            }
        )

        st.rerun()

    # --------------------------------------------------
    # Continue with normal RAG
    # --------------------------------------------------

    if not components:
        st.error(
            "Please build the knowledge base first. Add documents to "
            "`data/raw_docs/` and click **Build / Refresh Index**."
        )
        st.stop()

    try:
        llm = components["llm"]

        # Remove personal introduction before retrieval
        retrieval_question = remove_personal_fact_from_question(
            user_question
        )

        # Rewrite follow-up question
        if st.session_state.memory_enabled:
            history_text = format_history(
                st.session_state.turns
            )

            standalone_question = rewrite_question(
                llm,
                retrieval_question,
                history_text,
            )
        else:
            history_text = ""
            standalone_question = retrieval_question

        with st.status(
            "Searching the company knowledge base...",
            expanded=False,
        ) as status:

            candidates = hybrid_search(
                standalone_question,
                components["vectorstore"],
                components["chunks"],
                components["bm25"],
                top_k_vector=TOP_K_VECTOR,
                top_k_bm25=TOP_K_BM25,
                final_k=TOP_K_FINAL * 2,
            )

            status.update(
                label="Reranking relevant documents...",
                state="running",
            )

            reranked = components["reranker"].rerank(
                standalone_question,
                candidates,
                top_k=TOP_K_FINAL,
            )

            status.update(
                label="Generating a grounded answer...",
                state="running",
            )

            answer = generate_answer(
                llm,
                standalone_question,
                reranked,
                history_text=history_text,
            )

            if changed_facts:
                saved_items = ", ".join(
                    f"{key}: {value}"
                    for key, value in changed_facts.items()
                )

                answer = (
                    f"I saved this information: {saved_items}\n\n"
                    f"{answer}"
                )

            status.update(
                label="Answer ready",
                state="complete",
            )

        # Collect sources only when the answer is supported
        # by relevant retrieved context.

        if answer_has_no_relevant_information(answer):
            sources = []
            retrieved_for_display = []
        else:
            sources = []
            seen_sources = set()

            for item in reranked:
                label = source_label(item.get("metadata", {}))

                if label not in seen_sources:
                    sources.append(label)
                    seen_sources.add(label)

            retrieved_for_display = reranked

        st.session_state.turns.append(
            {
                "user": user_question,
                "assistant": answer,
                "sources": sources,
                "retrieved": retrieved_for_display,
                "rewritten_question": standalone_question,
            }
        )

        st.rerun()

    except Exception as error:
        st.error(
            f"Something went wrong while answering the question: {error}"
        )
