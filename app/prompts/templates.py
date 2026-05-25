from typing import List, Dict


SYSTEM_PROMPT = """You are a helpful, accurate, and concise customer support assistant.

Your ONLY source of truth is the context provided below. 
Do NOT use any prior knowledge outside the context.
If the context does not contain enough information to answer the question, clearly say so.
Keep your answers factual, clear, and polite.
"""


def build_history_block(history: List[Dict[str, str]]) -> str:
    """Format conversation history as a readable block."""
    if not history:
        return "No previous conversation."
    lines = []
    for turn in history:
        lines.append(f"User: {turn['user']}")
        lines.append(f"Assistant: {turn['assistant']}")
    return "\n".join(lines)


def build_prompt(
    retrieved_context: str,
    history: List[Dict[str, str]],
    user_question: str
) -> str:
    """
    Build the full LLM prompt with:
    - System instructions
    - Retrieved RAG context
    - Conversation history (last N turns)
    - Current user question
    """
    history_block = build_history_block(history)

    prompt = f"""{SYSTEM_PROMPT}

Context:
{retrieved_context}

Conversation History:
{history_block}

Question:
{user_question}

Answer:"""
    return prompt


FALLBACK_RESPONSE = "I could not find enough information in the knowledge base to answer this question."
