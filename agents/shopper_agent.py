"""
agents/shopper_agent.py
───────────────────────
Personal Shopper agent — tools + LangGraph graph + public runner.
Exports: run_personal_shopper(user_query: str) -> str
"""

import os
from dotenv import load_dotenv

from langchain_community.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from supabase_client import supabase, safe_tool_response

load_dotenv()

# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    request_timeout=30,
    max_retries=3,
)

# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def search_products(
    tags: str = None,
    max_price: float = None,
    size: str = None,
    is_sale: bool = None,
    is_clearance: bool = None,
    sort_by: str = None,
    vendor: str = None,
) -> str:
    """
    Search product_inventory table by any combination of filters.

    Args:
        tags: style/occasion keywords e.g. 'modest', 'evening', 'fitted', 'sleeve'
        max_price: maximum price in dollars e.g. 300
        size: size to check availability e.g. '8', 'M', 'XL'
        is_sale: set True to only return sale items
        is_clearance: set True to only return clearance items
        sort_by: 'bestseller_score' or 'price'
        vendor: brand or vendor name e.g. 'Lumiere', 'Nocturne', 'Aurelia Couture'
    """
    try:
        query = supabase.table("product_inventory").select(
            "product_id, title, vendor, price, compare_at_price, "
            "tags, sizes_available, is_sale, is_clearance, bestseller_score"
        )

        if vendor:
            query = query.ilike("vendor", f"%{vendor}%")
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
    Fetch full details of a single product from product_inventory.

    Args:
        product_id: the unique product ID string
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
    
def rank_products(products: list, size: str) -> list:
    """
    Rank by: in-stock for size → is_sale → bestseller_score → price.
    Called after get_product results are collected.
    """
    def score(p):
        stock = p.get("stock_per_size", {}).get(size, 0) if size else 1
        return (
            -(stock > 0),                          # in-stock first (negated so True sorts first)
            -int(p.get("is_sale", False)),          # sale items first
            -p.get("bestseller_score", 0),          # higher score first
            p.get("price", 9999),                   # lower price first
        )
    return sorted(products, key=score)


# ── System Prompt ─────────────────────────────────────────────────────────────
PERSONAL_SHOPPER_PROMPT = """You are an expert personal shopper for a fashion retail store.
Your job is to find the BEST product match for the customer's needs — not just any match.

You have access to these tools:
- search_products: search inventory with filters (tags, size, max_price, is_sale, is_clearance, sort_by, vendor)
- get_product: get full details of a specific product including stock_per_size

REASONING PROCESS — follow this every time:
1. Extract ALL constraints from the user query (size, budget, occasion, style, sale preference, vendor/brand name)
2. Call search_products with those constraints
3. For the top 2-3 results, call get_product to check stock_per_size for the requested size
4. Eliminate any product where stock for the requested size is 0
5. Among in-stock options, rank by: sale status first → bestseller_score second → price third
6. Recommend the SINGLE best match with full reasoning

RESPONSE FORMAT — always structure your response like this:
🛍️ **My Recommendation: [Product Title]**

**Why this fits you:**
- Size: [confirm size X is in stock — X units available]
- Budget: [$X — under your $Y budget / on sale from $Z]
- Style: [how tags match their occasion/style request]
- Popularity: [bestseller score context]

**Honest notes:**
- [Any tradeoff worth mentioning — e.g. low stock, only store credit on returns]

**Other options considered:** [briefly why you ranked them lower]

RULES:
- Never recommend a product where the requested size has 0 stock
- Always mention exactly how many units are left if stock is 3 or fewer (creates urgency)
- If nothing matches all constraints, relax one constraint and explain why
- Never hallucinate product details — only use data returned by tools
- If is_sale=True, mention the original compare_at_price vs current price
- Never generate function calls or XML syntax in your response
"""

# ── LLM with tools bound ──────────────────────────────────────────────────────
shopper_llm = llm.bind_tools([search_products, get_product])

# ── Graph nodes ───────────────────────────────────────────────────────────────

def personal_shopper_node(state: MessagesState):
    messages = [SystemMessage(content=PERSONAL_SHOPPER_PROMPT)] + state["messages"]
    response = shopper_llm.invoke(messages)
    return {"messages": [response]}



def should_continue(state: MessagesState):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

def ranker_node(state: MessagesState):
    """
    Intercepts tool results, extracts products, pre-ranks them,
    and injects ranking into the next LLM call as a system note.
    """
    from langchain_core.messages import ToolMessage
    import json

    # Collect all product data from tool messages in this turn
    products = []
    requested_size = None

    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage):
            try:
                data = json.loads(msg.content)
                # get_product returns a single dict; search_products returns a list
                if isinstance(data, dict) and "product_id" in data:
                    products.append(data)
                elif isinstance(data, list):
                    products.extend(data)
            except (json.JSONDecodeError, TypeError):
                pass

        # Extract size from the original human message
        if isinstance(msg, HumanMessage) and not requested_size:
            size_match = re.search(r'\bsize\s*(\w+)\b', msg.content, re.IGNORECASE)
            if size_match:
                requested_size = size_match.group(1)

    if not products:
        # No products to rank — pass through normally
        messages = [SystemMessage(content=PERSONAL_SHOPPER_PROMPT)] + state["messages"]
        response = shopper_llm.invoke(messages)
        return {"messages": [response]}

    ranked = rank_products(products, requested_size)
    ranking_note = (
        "\n\n[SYSTEM RANKING — follow this order strictly]\n"
        + "\n".join(
            f"#{i+1}: {p.get('title', 'Unknown')} | "
            f"sale={p.get('is_sale')} | "
            f"score={p.get('bestseller_score')} | "
            f"price=${p.get('price')} | "
            f"stock_size_{requested_size}="
            f"{p.get('stock_per_size', {}).get(requested_size, '?') if requested_size else 'n/a'}"
            for i, p in enumerate(ranked)
        )
        + "\nRecommend #1 unless it has 0 stock — then move to #2."
    )

    augmented_prompt = PERSONAL_SHOPPER_PROMPT + ranking_note
    messages = [SystemMessage(content=augmented_prompt)] + state["messages"]
    response = shopper_llm.invoke(messages)
    return {"messages": [response]}


_shopper_tools = [search_products, get_product]
_tool_node = ToolNode(_shopper_tools)

_shopper_graph = StateGraph(MessagesState)
_shopper_graph.add_node("personal_shopper", personal_shopper_node)
_shopper_graph.add_node("tools", _tool_node)
_shopper_graph.add_node("ranker", ranker_node)          # ← new node

_shopper_graph.add_edge(START, "personal_shopper")
_shopper_graph.add_conditional_edges(
    "personal_shopper",
    should_continue,
    {"tools": "tools", END: END}
)
_shopper_graph.add_edge("tools", "ranker")              # ← tools → ranker, not back to shopper
_shopper_graph.add_edge("ranker", END)                  # ← ranker always ends

shopper_agent = _shopper_graph.compile()



# ── Internal helpers ──────────────────────────────────────────────────────────

def _simplify_query(query: str) -> str:
    """Strip product-name tokens that confuse Groq's tool-call generation."""
    result = llm.invoke([
        SystemMessage(content=(
            "Rewrite the user's shopping query as a plain, simple sentence. "
            "Keep ONLY: size, budget, occasion, and style words. "
            "Remove any brand names, product names, or model numbers. "
            "Output only the rewritten query, nothing else."
        )),
        HumanMessage(content=query),
    ])
    return result.content.strip()


def _invoke_shopper(query: str) -> str:
    result = shopper_agent.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"recursion_limit": 10},
    )
    last = result["messages"][-1].content
    if "tool_use_failed" in str(last) or "failed_generation" in str(last):
        raise ValueError("tool_use_failed detected in response")
    return last


# ── Public runner ─────────────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
)
def run_personal_shopper(user_query: str) -> str:
    """
    Public entry point used by orchestrator.py and app.py.
    Handles Groq tool-call failures with automatic query simplification.
    """
    try:
        return _invoke_shopper(user_query)

    except Exception as e:
        err = str(e)

        if "tool_use_failed" in err or "failed_generation" in err:
            print("⚠️  Malformed tool call — simplifying query and retrying...")
            try:
                return _invoke_shopper(_simplify_query(user_query))
            except Exception:
                return (
                    "I had trouble searching right now. "
                    "Please try rephrasing — for example: "
                    "'Show me sale dresses under $150 in size 8.'"
                )

        elif "disconnected" in err.lower():
            print("⚠️  Groq disconnected — retrying...")
            raise

        elif "rate_limit" in err.lower():
            print("⚠️  Rate limited — waiting...")
            raise

        else:
            return f"Agent error: {err}"