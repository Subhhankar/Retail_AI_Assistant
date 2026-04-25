"""
agents/support_agent.py
───────────────────────
Customer Support agent — tools + LangGraph graph + public runner.
Exports: run_customer_support(user_query: str) -> str
"""

import os
from datetime import datetime
from dotenv import load_dotenv
import re
import json

from langchain_community.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from tenacity import retry, stop_after_attempt, wait_exponential

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
def get_order(order_id: str) -> str:
    """
    Fetch a single order by order_id including linked product details.

    Args:
        order_id: the unique order ID string e.g. 'O0002'
    """
    try:
        order_id = order_id.strip()

        result = (
            supabase.table("orders")
            .select(
                "order_id, order_date, size, price_paid, customer_id, "
                "product_inventory(product_id, title, vendor, price, "
                "is_sale, is_clearance, tags)"
            )
            .eq("order_id", order_id.upper())
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

    Args:
        order_id: the unique order ID string e.g. 'O0002'
    """
    # ── Vendor-specific overrides ──────────────────────────────────────────
    VENDOR_RULES = {
        "aurelia couture": {
            "return_type": "exchange_only",
            "window_days": 14,
            "note": "Aurelia Couture: exchanges only, no refunds"
        },
        "nocturne": {
            "return_type": "full_refund",
            "window_days": 21,
            "note": "Nocturne: extended 21-day return window"
        },
    }

    try:
        order_id = order_id.strip()

        result = (
            supabase.table("orders")
            .select(
                "order_id, order_date, size, price_paid, customer_id, "
                "product_inventory(title, vendor, is_sale, is_clearance)"
            )
            .eq("order_id", order_id.upper())
            .limit(1)
            .execute()
        )

        if not result.data:
            return "No order found for that order ID."

        order   = result.data[0]
        product = order.get("product_inventory", {})
        vendor  = (product.get("vendor") or "").lower().strip()

        order_date = datetime.fromisoformat(order["order_date"])
        days_since = (datetime.now() - order_date).days

        # 1. Clearance — always final sale, no exceptions
        if product.get("is_clearance"):
            verdict = {
                "eligible": False,
                "return_type": None,
                "reason": "Clearance items are final sale — not eligible for return or exchange",
                "days_since_order": days_since,
                "order": order,
            }

        # 2. Vendor exception — Aurelia Couture (exchange only, 14-day window)
        elif vendor in VENDOR_RULES and vendor == "aurelia couture":
            rule = VENDOR_RULES[vendor]
            if days_since > rule["window_days"]:
                verdict = {
                    "eligible": False,
                    "return_type": None,
                    "reason": f"Aurelia Couture exchange window expired — order was {days_since} days ago (limit: {rule['window_days']} days)",
                    "days_since_order": days_since,
                    "order": order,
                }
            else:
                verdict = {
                    "eligible": True,
                    "return_type": "exchange_only",
                    "reason": f"Aurelia Couture policy: exchanges only, no refunds. {rule['window_days'] - days_since} days remaining",
                    "days_since_order": days_since,
                    "order": order,
                    "exchange_note": "Size exchanges allowed if stock available. Customer pays return shipping unless item is defective.",
                }

        # 3. Vendor exception — Nocturne (extended 21-day window, full refund)
        elif vendor in VENDOR_RULES and vendor == "nocturne":
            rule = VENDOR_RULES[vendor]
            if days_since > rule["window_days"]:
                verdict = {
                    "eligible": False,
                    "return_type": None,
                    "reason": f"Nocturne return window expired — order was {days_since} days ago (extended limit: {rule['window_days']} days)",
                    "days_since_order": days_since,
                    "order": order,
                }
            else:
                verdict = {
                    "eligible": True,
                    "return_type": "full_refund",
                    "reason": f"Nocturne extended return window: eligible for full refund. {rule['window_days'] - days_since} days remaining",
                    "days_since_order": days_since,
                    "order": order,
                }

        # 4. Sale items — 7-day window, store credit only
        elif product.get("is_sale"):
            if days_since > 7:
                verdict = {
                    "eligible": False,
                    "return_type": None,
                    "reason": f"Sale item return window expired — order was {days_since} days ago (limit: 7 days)",
                    "days_since_order": days_since,
                    "order": order,
                }
            else:
                verdict = {
                    "eligible": True,
                    "return_type": "store_credit",
                    "reason": f"Sale items qualify for store credit only. {7 - days_since} days remaining",
                    "days_since_order": days_since,
                    "order": order,
                }

        # 5. Normal items — 14-day window, full refund
        elif days_since > 14:
            verdict = {
                "eligible": False,
                "return_type": None,
                "reason": f"Return window expired — order was {days_since} days ago (limit: 14 days)",
                "days_since_order": days_since,
                "order": order,
            }
        else:
            verdict = {
                "eligible": True,
                "return_type": "full_refund",
                "reason": f"Eligible for full refund — {14 - days_since} days remaining",
                "days_since_order": days_since,
                "order": order,
                "shipping_note": "Customer pays return shipping unless item is defective.",
            }

        return safe_tool_response(verdict)

    except Exception as e:
        return f"Return evaluation failed: {str(e)}"


# ── System Prompt ─────────────────────────────────────────────────────────────
CUSTOMER_SUPPORT_PROMPT = """You are a Customer Support Assistant for a fashion retail store.
Your job is to answer policy questions and handle order-specific return requests.

You have access to these tools:
- get_order: fetch order details and linked product info by order_id
- evaluate_return: apply return policy rules and get eligibility decision

RETURN POLICY — know this exactly:

Normal Items:
- Returns accepted within 14 days of delivery
- Eligible for full refund
- Customer pays return shipping unless item is defective

Sale Items:
- Returnable within 7 days only
- Store credit only — no cash refund

Clearance Items:
- Final sale — NOT eligible for return or exchange under any circumstances

Vendor Exceptions (override standard rules):
- Aurelia Couture: Exchanges only, no refunds. 14-day window.
- Nocturne: Extended return window of 21 days. Full refund eligible.

Exchange Rules:
- Size exchanges allowed if replacement size is in stock
- Customer pays return shipping unless item is defective

─────────────────────────────────────────────
STEP 1 — CLASSIFY THE REQUEST BEFORE DOING ANYTHING:

A) GENERAL POLICY QUESTION — the message asks about policy in general,
   with NO specific order ID mentioned (e.g. "what is the return window?",
   "how many days do I have to return?", "what is your refund policy?").
   → Answer directly from the policy above. Do NOT call any tools.
   → Never reference a previous order from earlier in the conversation.

B) ORDER-SPECIFIC REQUEST — the message contains an order ID (e.g. O0002)
   OR explicitly asks about a specific order.
   → Call get_order then evaluate_return, then give the decision.
─────────────────────────────────────────────

RESPONSE FORMAT FOR GENERAL POLICY QUESTIONS:
📋 **Return Policy Summary**

- **Normal items:** Full refund within 14 days. Customer pays return shipping unless defective.
- **Sale items:** Store credit only, within 7 days.
- **Clearance items:** Final sale — no returns or exchanges.
- **Aurelia Couture:** Exchanges only (no refunds), within 14 days.
- **Nocturne:** Full refund within 21 days.
- **Size exchanges:** Allowed if replacement size is in stock.

RESPONSE FORMAT FOR ORDER-SPECIFIC REQUESTS:
📦 **Order #[order_id] — [Product Title]**

**Purchase details:**
- Ordered: [order_date] ([X] days ago)
- Size purchased: [size]
- Amount paid: $[price_paid]
- Vendor: [vendor]

**Return decision: ✅ Approved / ❌ Not Eligible**

**Reason:**
[Clear explanation referencing the exact policy rule that applies.]

**Next steps:**
[Exact instructions based on decision.]

RULES:
- NEVER call tools for general policy questions — answer from policy knowledge directly
- NEVER pull a previous order into a new general question
- Never make up order details — only use data from tools
- Never override policy rules even if the customer is upset
- Always be empathetic but firm on policy
- If order_id is not found, ask the customer to verify their order number
"""

def extract_order_id(text: str) -> str | None:
    match = re.search(r'\bO\d{4}\b', text.upper())
    return match.group(0) if match else None

# ── LLM with tools bound ──────────────────────────────────────────────────────
support_llm = llm.bind_tools([get_order, evaluate_return])

# ── Graph nodes ───────────────────────────────────────────────────────────────

def customer_support_node(state: MessagesState):
    from langchain_core.messages import ToolMessage

    last_user_msg = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None
    )

    # Fix 1 — validate order ID first
    if last_user_msg:
        order_id = extract_order_id(last_user_msg.content)
        if not order_id:
            return {"messages": [AIMessage(
                content=(
                    "I'd be happy to help with your order. "
                    "Could you please provide your order ID? "
                    "It should be in the format **O0001**, **O0002**, etc. "
                    "You can find it in your confirmation email."
                )
            )]}

    # Fix 3 — if tool results exist, inject raw values to prevent formatting drift
    verdict_note = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage):
            try:
                data = json.loads(msg.content)
                if "eligible" in data:  # this is an evaluate_return result
                    verdict_note = (
                        "\n\n[SYSTEM — use these exact values, do not paraphrase]\n"
                        f"eligible: {data['eligible']}\n"
                        f"return_type: {data.get('return_type')}\n"
                        f"reason: {data.get('reason')}\n"
                        f"days_since_order: {data.get('days_since_order')}\n"
                        f"price_paid: {data.get('order', {}).get('price_paid')}\n"
                        f"order_date: {data.get('order', {}).get('order_date')}\n"
                        f"size: {data.get('order', {}).get('size')}\n"
                    )
            except (json.JSONDecodeError, TypeError):
                pass
        break  # only check the most recent tool message

    augmented_prompt = CUSTOMER_SUPPORT_PROMPT + verdict_note
    messages = [SystemMessage(content=augmented_prompt)] + state["messages"]
    response = support_llm.invoke(messages)
    return {"messages": [response]}




def support_should_continue(state: MessagesState):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# ── Build graph ───────────────────────────────────────────────────────────────
_support_tools = [get_order, evaluate_return]
_support_tool_node = ToolNode(_support_tools)

_support_graph = StateGraph(MessagesState)
_support_graph.add_node("customer_support", customer_support_node)
_support_graph.add_node("tools", _support_tool_node)

_support_graph.add_edge(START, "customer_support")
_support_graph.add_conditional_edges(
    "customer_support",
    support_should_continue,
    {"tools": "tools", END: END},
)
_support_graph.add_edge("tools", "customer_support")

support_agent = _support_graph.compile()


# ── Public runner ─────────────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def run_customer_support(user_query: str) -> str:
    """
    Public entry point used by orchestrator.py and app.py.
    """
    try:
        result = support_agent.invoke(
            {"messages": [HumanMessage(content=user_query)]},
            config={"recursion_limit": 10},
        )
        return result["messages"][-1].content

    except Exception as e:
        return f"Agent error: {str(e)}"