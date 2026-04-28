"""
agent.py
────────
Single unified agent for Maison AI Retail Assistant.
One brain, all 4 tools — handles shopping AND support in one graph.
Exports: run_agent(user_query: str, chat_history: list) -> str
"""

import os
from datetime import datetime
from dotenv import load_dotenv

from langchain_community.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from supabase_client import supabase, safe_tool_response

load_dotenv()

# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    request_timeout=30,
    max_retries=3,
)

# ══════════════════════════════════════════════════════════════════════════════
# TOOLS
# ══════════════════════════════════════════════════════════════════════════════

@tool
def search_products(
    tags: str = None,
    max_price: float = None,
    size: str = None,
    is_sale: bool = None,
    is_clearance: bool = None,
    sort_by: str = None,
) -> str:
    """
    Search product_inventory by any combination of filters.

    Args:
        tags: style/occasion keywords e.g. 'modest', 'evening', 'fitted'
        max_price: maximum price in dollars e.g. 300
        size: size to check availability e.g. '8', 'M', 'XL'
        is_sale: True to return only sale items
        is_clearance: True to return only clearance items
        sort_by: 'bestseller_score' or 'price'
    """
    try:
        query = supabase.table("product_inventory").select(
            "product_id, title, price, compare_at_price, "
            "tags, sizes_available, is_sale, is_clearance, bestseller_score"
        )
        if max_price is not None:
            query = query.lte("price", int(max_price))
        if is_sale is True:
            query = query.eq("is_sale", True)
        if is_clearance is True:
            query = query.eq("is_clearance", True)
        if tags:
            query = query.ilike("tags", f"%{tags}%")
        if size:
            query = query.ilike("sizes_available", f"%{size}%")
        if sort_by == "bestseller_score":
            query = query.order("bestseller_score", desc=True)
        elif sort_by == "price":
            query = query.order("price", desc=False)

        result = query.limit(5).execute()
        return safe_tool_response(result.data)

    except Exception as e:
        return f"Search failed: {str(e)}"


@tool
def get_product(product_id: str) -> str:
    """
    Fetch full details of a single product including stock_per_size.

    Args:
        product_id: unique product ID string e.g. 'P0042'
    """
    try:
        result = (
            supabase.table("product_inventory")
            .select("*")
            .eq("product_id", product_id)
            .limit(1)
            .execute()
        )
        return safe_tool_response(result.data[0] if result.data else None)
    except Exception as e:
        return f"Product lookup failed: {str(e)}"


@tool
def get_order(order_id: str) -> str:
    """
    Fetch a single order by order_id including linked product details.

    Args:
        order_id: unique order ID string e.g. 'O0002'
    """
    try:
        order_id = order_id.strip().upper()
        result = (
            supabase.table("orders")
            .select(
                "order_id, order_date, size, price_paid, customer_id, "
                "product_inventory(product_id, title, vendor, price, "
                "is_sale, is_clearance, tags)"
            )
            .eq("order_id", order_id)
            .limit(1)
            .execute()
        )
        return safe_tool_response(result.data[0] if result.data else None)
    except Exception as e:
        return f"Order lookup failed: {str(e)}"


@tool
def evaluate_return(order_id: str) -> str:
    """
    Evaluate return eligibility for an order based on store policy.

    Policy rules:
    - Clearance items → never returnable
    - Aurelia Couture → exchange only, no refunds
    - Sale items → store credit within 7 days only
    - Nocturne → full refund within 21 days
    - All other items → full refund within 14 days

    Args:
        order_id: unique order ID string e.g. 'O0002'
    """
    try:
        order_id = order_id.strip().upper()
        result = (
            supabase.table("orders")
            .select(
                "order_id, order_date, size, price_paid, customer_id, "
                "product_inventory(title, vendor, is_sale, is_clearance)"
            )
            .eq("order_id", order_id)
            .limit(1)
            .execute()
        )

        if not result.data:
            return "No order found for that order ID."

        order   = result.data[0]
        product = order.get("product_inventory", {})
        vendor  = product.get("vendor", "")

        order_date = datetime.fromisoformat(order["order_date"])
        days_since = (datetime.now() - order_date).days

        if product.get("is_clearance"):
            verdict = {
                "eligible": False, "return_type": None,
                "reason": "Clearance items are final sale — not returnable",
                "days_since_order": days_since, "order": order,
            }

        elif vendor == "Aurelia Couture":
            verdict = {
                "eligible": True, "return_type": "exchange_only",
                "reason": "Aurelia Couture policy — exchanges only, no refunds. Customer pays return shipping unless defective.",
                "days_since_order": days_since, "order": order,
            }

        elif product.get("is_sale") and days_since <= 7:
            verdict = {
                "eligible": True, "return_type": "store_credit",
                "reason": f"Sale item — store credit only. {7 - days_since} days remaining in return window.",
                "days_since_order": days_since, "order": order,
            }

        elif product.get("is_sale") and days_since > 7:
            verdict = {
                "eligible": False, "return_type": None,
                "reason": f"Sale item return window expired — only 7 days allowed, order was {days_since} days ago.",
                "days_since_order": days_since, "order": order,
            }

        elif vendor == "Nocturne" and days_since <= 21:
            verdict = {
                "eligible": True, "return_type": "full_refund",
                "reason": f"Nocturne extended return window — {21 - days_since} days remaining.",
                "days_since_order": days_since, "order": order,
            }

        elif vendor == "Nocturne" and days_since > 21:
            verdict = {
                "eligible": False, "return_type": None,
                "reason": f"Nocturne extended window expired — 21 days allowed, order was {days_since} days ago.",
                "days_since_order": days_since, "order": order,
            }

        elif days_since <= 14:
            verdict = {
                "eligible": True, "return_type": "full_refund",
                "reason": f"Eligible for full refund — {14 - days_since} days remaining in return window.",
                "days_since_order": days_since, "order": order,
            }

        else:
            verdict = {
                "eligible": False, "return_type": None,
                "reason": f"Return window expired — 14 days allowed, order was {days_since} days ago.",
                "days_since_order": days_since, "order": order,
            }

        return safe_tool_response(verdict)

    except Exception as e:
        return f"Return evaluation failed: {str(e)}"


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are Maison, an intelligent AI assistant for a premium fashion retail store.
You handle everything — product discovery, recommendations, order inquiries, and return requests.

You have access to four tools:
- search_products: search inventory by tags, size, price, sale/clearance status
- get_product: get full product details including per-size stock levels
- get_order: fetch order details and linked product info by order_id
- evaluate_return: apply return policy and get eligibility verdict

━━━ WHEN SHOPPING ━━━
1. Extract all constraints (size, budget, occasion, style, sale preference)
2. Call search_products with those constraints
3. For top 2-3 results call get_product to check stock for the requested size
4. Eliminate products where stock for the requested size is 0
5. Rank by: sale status → bestseller_score → price
6. Recommend the single best match with clear reasoning

Response format for shopping:
🛍️ **My Recommendation: [Product Title]**
**Why this fits you:**
- Size: [confirm size X is in stock — X units available]
- Budget: [$X — under your $Y budget / on sale from $Z]
- Style: [how tags match their request]
- Popularity: [bestseller score context]
**Honest notes:** [tradeoffs, low stock urgency, return policy note]
**Other options considered:** [briefly why ranked lower]

━━━ WHEN HANDLING ORDERS / RETURNS ━━━
1. Extract the order_id from the message
2. Call get_order to fetch order and product details
3. Call evaluate_return to apply policy rules
4. Give a clear approved / not eligible decision

Response format for returns:
📦 **Order #[order_id] — [Product Title]**
**Purchase details:**
- Ordered: [date] ([X] days ago)
- Size: [size] | Amount paid: $[price_paid]
**Return decision: ✅ Approved / ❌ Not Eligible**
**Reason:** [2-3 sentences, policy-based, never guess]
**Next steps:** [exact instructions or alternatives]
POLICY RULES:
- Clearance → never returnable
- Normal items → full refund within 14 days
- Sale items → store credit within 7 days only
- Aurelia Couture → exchange only, no refunds
- Nocturne → extended 21-day return window
- Size exchanges allowed if stock is available
- Customer pays return shipping unless item is defective

━━━ RULES ━━━
- Never hallucinate — only use data from tools
- Never recommend out-of-stock sizes
- Never override return policy rules
- If stock is 3 or fewer, mention exact count to create urgency
- If order_id not found, ask the customer to verify it
- Always mention compare_at_price vs sale price when is_sale=True
- Never generate XML, JSON, or function call syntax in your replies
"""

# ── LLM with all tools bound ──────────────────────────────────────────────────
ALL_TOOLS = [search_products, get_product, get_order, evaluate_return]
agent_llm = llm.bind_tools(ALL_TOOLS)


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH
# ══════════════════════════════════════════════════════════════════════════════

def agent_node(state: MessagesState):
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = agent_llm.invoke(messages)
    return {"messages": [response]}


def should_continue(state: MessagesState):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


_tool_node = ToolNode(ALL_TOOLS)

_graph = StateGraph(MessagesState)
_graph.add_node("agent", agent_node)
_graph.add_node("tools", _tool_node)

_graph.add_edge(START, "agent")
_graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
_graph.add_edge("tools", "agent")

maison_agent = _graph.compile()


# ══════════════════════════════════════════════════════════════════════════════
# QUERY SIMPLIFIER  (Groq tool-call failure recovery)
# ══════════════════════════════════════════════════════════════════════════════

def _simplify_query(query: str) -> str:
    result = llm.invoke([
        SystemMessage(content=(
            "Rewrite the shopping query as a plain simple sentence. "
            "Keep ONLY: size, budget, occasion, style words. "
            "Remove brand names, product names, model numbers. "
            "Output only the rewritten query, nothing else."
        )),
        HumanMessage(content=query),
    ])
    return result.content.strip()


def _invoke(messages: list) -> str:
    result = maison_agent.invoke(
        {"messages": messages},
        config={"recursion_limit": 12},
    )
    last = result["messages"][-1].content
    if "tool_use_failed" in str(last) or "failed_generation" in str(last):
        raise ValueError("tool_use_failed")
    return last


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC RUNNER
# ══════════════════════════════════════════════════════════════════════════════

# Module-level chat history — synced by app.py via session state
chat_history: list = []


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
)
def run_agent(user_query: str) -> str:
    """
    Main entry point called by app.py.
    Maintains conversation history internally.
    """
    global chat_history

    chat_history.append(HumanMessage(content=user_query))

    try:
        response = _invoke(chat_history)

    except Exception as e:
        err = str(e)

        if "tool_use_failed" in err or "failed_generation" in err:
            print("⚠️  Malformed tool call — simplifying and retrying...")
            try:
                simplified_history = chat_history[:-1] + [
                    HumanMessage(content=_simplify_query(user_query))
                ]
                response = _invoke(simplified_history)
            except Exception:
                response = (
                    "I had a little trouble with that request. "
                    "Could you rephrase it? For example: "
                    "'Show me sale dresses under $150 in size 8' "
                    "or 'I want to return order O0002'."
                )

        elif "disconnected" in err.lower() or "rate_limit" in err.lower():
            raise  # let tenacity retry

        else:
            response = f"Something went wrong: {err}"

    chat_history.append(AIMessage(content=response))
    return response


def reset_history() -> None:
    """Clear conversation history — called by Streamlit's Clear button."""
    global chat_history
    chat_history = []
