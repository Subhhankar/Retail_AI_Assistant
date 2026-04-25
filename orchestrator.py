"""
orchestrator.py
───────────────
Central orchestration layer for the Maison AI Retail Assistant.
This file is imported by app.py and wires together:
  • Personal Shopper agent  (search_products, get_product)
  • Customer Support agent  (get_order, evaluate_return)

Drop this file next to app.py.
All your existing agent/tool/graph cells should be in a separate
notebook or imported here after converting to .py modules.
"""

import os
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ── Load env ──────────────────────────────────────────────────────────────────
load_dotenv()

# ── Import agents ─────────────────────────────────────────────────────────────
from agents.shopper_agent import run_personal_shopper
from agents.support_agent import run_customer_support

# ── LLM (used only for routing classification) ────────────────────────────────
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0,          # deterministic routing
    request_timeout=30,
    max_retries=3,
)

# ── Orchestrator prompt ───────────────────────────────────────────────────────
ORCHESTRATOR_PROMPT = """You are a routing orchestrator for a fashion retail store assistant.
Your ONLY job is to classify the user's intent and respond with exactly one word.

Respond with ONLY:
- "shopper"  → if the user wants to browse, search, find, or get recommendations for products
- "support"  → if the user mentions an order ID, return, refund, or order status

Examples:
"I want a red dress under $50" → shopper
"Order O0002 doesn't fit, can I return it?" → support
"Show me sale items" → shopper
"What's the status of my order?" → support
"Can I return order O0015?" → support
"Find me something for a wedding" → shopper

Respond with ONLY the single word. No punctuation, no explanation.
Never generate function calls, JSON, or XML. One word only."""

# ── Conversation history (module-level so Streamlit can sync it) ──────────────
chat_history: list = []


# ── Internal helpers ──────────────────────────────────────────────────────────

def _classify_intent(query: str) -> str:
    """Returns 'shopper' or 'support'."""
    try:
        result = llm.invoke([
            SystemMessage(content=ORCHESTRATOR_PROMPT),
            HumanMessage(content=query),
        ])
        intent = result.content.strip().lower()
        return intent if intent in ("shopper", "support") else "shopper"
    except Exception:
        return "shopper"  # safe default


def _build_shopper_context(query: str) -> str:
    """Prepend recent history to give the shopper agent conversation context."""
    recent = chat_history[-4:]  # last 4 messages
    context_lines = []
    for m in recent:
        if isinstance(m, HumanMessage):
            context_lines.append(f"Customer: {m.content}")
        elif isinstance(m, AIMessage):
            context_lines.append(f"Assistant: {m.content}")

    if context_lines:
        return (
            "Conversation so far:\n"
            + "\n".join(context_lines)
            + f"\n\nCurrent request: {query}"
        )
    return query


def _safe_shopper(query: str) -> str:
    try:
        response = run_personal_shopper(_build_shopper_context(query))
    except Exception as e:
        response = f"Agent error: {str(e)}"

    if response.startswith("Agent error:") or "tool_use_failed" in response:
        # Retry with simplified query
        try:
            simplified = llm.invoke([
                SystemMessage(content=(
                    "Rewrite the user's shopping query as a plain, simple sentence. "
                    "Keep ONLY: size, budget, occasion, and style words. "
                    "Remove any brand names, product names, or model numbers. "
                    "Output only the rewritten query, nothing else."
                )),
                HumanMessage(content=query),
            ]).content.strip()
            response = run_personal_shopper(simplified)
        except Exception:
            response = (
                "I had trouble searching right now. "
                "Please try rephrasing — for example: "
                "'Show me sale dresses under $150 in size 8.'"
            )

    return response


def _safe_support(query: str) -> str:
    try:
        response = run_customer_support(query)
    except Exception as e:
        response = f"Agent error: {str(e)}"

    if response.startswith("Agent error:"):
        response = (
            "I couldn't retrieve your order details. "
            "Please verify your order ID (e.g. O0002) and try again."
        )

    return response


# ── Public API ────────────────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
)
def run_orchestrator(user_query: str) -> str:
    """
    Main entry point called by app.py.
    Classifies intent, routes to the correct agent, updates chat_history.
    """
    global chat_history

    chat_history.append(HumanMessage(content=user_query))

    intent = _classify_intent(user_query)

    if intent == "support":
        response = _safe_support(user_query)
    else:
        response = _safe_shopper(user_query)

    chat_history.append(AIMessage(content=response))
    return response


def reset_history() -> None:
    """Call this to clear conversation history (e.g. from Streamlit's Clear button)."""
    global chat_history
    chat_history = []