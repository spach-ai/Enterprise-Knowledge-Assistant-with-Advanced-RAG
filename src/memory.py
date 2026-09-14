import json
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from src.config import DATA_DIR
from src.utils import ensure_dir

MEMORY_FILE = DATA_DIR / "user_memory.json"

DEFAULT_MEMORY = {
    "facts": {},
    "updated_at": None,
}


# -----------------------------
# Persistence helpers
# -----------------------------
def load_memory() -> Dict:
    """
    Load persistent memory from JSON.
    Returns a dict with a 'facts' key.
    """
    ensure_dir(DATA_DIR)

    if not MEMORY_FILE.exists():
        return deepcopy(DEFAULT_MEMORY)

    try:
        with MEMORY_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return deepcopy(DEFAULT_MEMORY)

        if "facts" not in data or not isinstance(data["facts"], dict):
            data["facts"] = {}

        return data
    except Exception:
        return deepcopy(DEFAULT_MEMORY)


def save_memory(memory: Dict) -> None:
    """
    Save memory to JSON.
    """
    ensure_dir(DATA_DIR)
    memory["updated_at"] = datetime.utcnow().isoformat()

    with MEMORY_FILE.open("w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)


def clear_memory() -> None:
    """
    Delete persistent memory file.
    """
    if MEMORY_FILE.exists():
        MEMORY_FILE.unlink()


# -----------------------------
# Chat history formatting
# -----------------------------
def format_history(turns: List[Dict], max_turns: int = 6) -> str:
    if not turns:
        return ""

    selected = turns[-max_turns:]
    lines = []
    for turn in selected:
        lines.append(f"User: {turn['user']}")
        lines.append(f"Assistant: {turn['assistant']}")
    return "\n".join(lines)


def rewrite_question(llm, question: str, history_text: str) -> str:
    """
    Rewrites follow-up questions into standalone questions using recent chat history.
    This is for retrieval context, not for persistent memory.
    """
    if not history_text.strip():
        return question.strip()

    from langchain_core.messages import HumanMessage, SystemMessage

    system_prompt = (
        "You rewrite follow-up questions into standalone questions.\n"
        "Use the conversation history to resolve pronouns and context.\n"
        "Return ONLY the rewritten question. Do not add explanations."
    )

    user_prompt = f"""
Conversation history:
{history_text}

User question:
{question}

Standalone rewritten question:
""".strip()

    try:
        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        rewritten = response.content.strip()
        return rewritten if rewritten else question.strip()
    except Exception:
        return question.strip()


# -----------------------------
# Fact extraction
# -----------------------------
def _first_match(patterns: List[str], text: str) -> Optional[str]:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)

        if match:
            value = match.group(1).strip()
            value = value.rstrip(".,!?")
            return value

    return None


def extract_facts(text: str) -> Dict[str, str]:
    """
    Extract structured facts from user input.

    Supported facts:
    - name
    - department
    - role
    - location
    - manager
    - email
    """
    text = text.strip()

    fact_patterns = {
        "name": [
            r"\bmy name is\s+([A-Za-z][A-Za-z\s\-']{0,50}?)(?=\s+(?:what|who|where|how|and|but|or)\b|[,.!?]|$)",
            r"\bcall me\s+([A-Za-z][A-Za-z\s\-']{0,50}?)(?=\s+(?:what|who|where|how|and|but|or)\b|[,.!?]|$)",
        ],
        "department": [
            r"\bmy department is\s+([A-Za-z][A-Za-z0-9\s&/\-']{0,50})",
            r"\bi work in\s+([A-Za-z][A-Za-z0-9\s&/\-']{0,50})",
            r"\bi work(?:s)? in the\s+([A-Za-z][A-Za-z0-9\s&/\-']{0,50})",
            r"\bi am in\s+([A-Za-z][A-Za-z0-9\s&/\-']{0,50})\s+department",
        ],
        "role": [
            r"\bmy role is\s+([A-Za-z][A-Za-z0-9\s&/\-']{0,70})",
            r"\bmy title is\s+([A-Za-z][A-Za-z0-9\s&/\-']{0,70})",
            r"\bi work as\s+([A-Za-z][A-Za-z0-9\s&/\-']{0,70})",
            r"\bi am a[n]?\s+([A-Za-z][A-Za-z0-9\s&/\-']{0,70})",
        ],
        "location": [
            r"\bi am from\s+([A-Za-z][A-Za-z0-9\s&/\-']{0,70})",
            r"\bmy location is\s+([A-Za-z][A-Za-z0-9\s&/\-']{0,70})",
            r"\bi live in\s+([A-Za-z][A-Za-z0-9\s&/\-']{0,70})",
            r"\bi am based in\s+([A-Za-z][A-Za-z0-9\s&/\-']{0,70})",
        ],
        "manager": [
            r"\bmy manager is\s+([A-Za-z][A-Za-z\s\-']{0,70})",
            r"\bi report to\s+([A-Za-z][A-Za-z\s\-']{0,70})",
            r"\bmy supervisor is\s+([A-Za-z][A-Za-z\s\-']{0,70})",
        ],
        "email": [
            r"\bmy email is\s+([^\s,;]+@[^\s,;]+)",
            r"\byou can email me at\s+([^\s,;]+@[^\s,;]+)",
            r"\bcontact me at\s+([^\s,;]+@[^\s,;]+)",
        ],
    }

    facts = {}
    for key, patterns in fact_patterns.items():
        value = _first_match(patterns, text)
        if value:
            facts[key] = value

    return facts


def update_memory_from_text(text: str, memory: Dict) -> Dict[str, str]:
    """
    Extract facts from text and store them in persistent memory.
    Returns the facts that were added/updated.
    """
    new_facts = extract_facts(text)
    if not new_facts:
        return {}

    if "facts" not in memory or not isinstance(memory["facts"], dict):
        memory["facts"] = {}

    changed = {}
    for key, value in new_facts.items():
        old_value = memory["facts"].get(key)
        if old_value != value:
            memory["facts"][key] = value
            changed[key] = value

    if changed:
        save_memory(memory)

    return changed


# -----------------------------
# Memory question answering
# -----------------------------
def normalize_question(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def get_memory_intent(question: str) -> Optional[str]:
    q = normalize_question(question)

    intents = {
        "name": [
            "what is my name",
            "whats my name",
            "what's my name",
            "do you know my name",
            "tell me my name",
            "remember my name",
            "who am i",
        ],
        "department": [
            "what is my department",
            "which department do i work in",
            "what department do i work in",
            "what team do i work in",
        ],
        "role": [
            "what is my role",
            "what is my title",
            "what do i do",
            "what job do i have",
        ],
        "location": [
            "where am i based",
            "what is my location",
            "where do i live",
        ],
        "manager": [
            "who is my manager",
            "what is my manager",
            "what is my manager name",
            "what is my manager's name",
            "do you know my manager",
            "who do i report to",
            "what is my supervisor",
            "who is my supervisor",
        ],
        "email": [
            "what is my email",
            "what email do you have for me",
        ],
        "summary": [
            "what do you know about me",
            "tell me what you know about me",
            "remember what i told you",
            "show my profile",
        ],
    }

    for intent, phrases in intents.items():
        for phrase in phrases:
            if phrase in q:
                return intent

    return None


def answer_from_memory(question: str, memory: Dict) -> Optional[str]:
    """
    Generate a response directly from persistent memory if possible.
    """
    intent = get_memory_intent(question)
    if not intent:
        return None

    facts = memory.get("facts", {})

    if intent == "summary":
        if not facts:
            return "I don't remember any personal details yet."
        lines = ["Here is what I remember about you:"]
        for key, value in facts.items():
            pretty_key = key.replace("_", " ").title()
            lines.append(f"- {pretty_key}: {value}")
        return "\n".join(lines)

    value = facts.get(intent)

    if value:
        if intent == "manager":
            return f"Your manager's name is {value}."

        if intent == "email":
            return f"Your email address is {value}."

        return f"Your {intent} is {value}."

    return (
        f"I don't know your {intent} yet. "
        f"You can tell me by saying something like 'My {intent} is ...'."
    )

# -----------------------------
# Convenience helper
# -----------------------------
def get_memory_summary(memory: Dict) -> str:
    facts = memory.get("facts", {})
    if not facts:
        return "No saved facts."

    lines = []
    for key, value in facts.items():
        lines.append(f"{key}: {value}")
    return "\n".join(lines)