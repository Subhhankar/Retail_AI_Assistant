import sys
import os
import re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(
    page_title="MAISON — AI Retail Assistant",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300&family=Montserrat:wght@300;400;500;600&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #0a0a0a !important;
    color: #f0ece4 !important;
    font-family: 'Montserrat', sans-serif !important;
}
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 40% at 50% 0%, rgba(180,150,100,0.07) 0%, transparent 70%),
        #0a0a0a !important;
}
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0e0e0e !important;
    border-right: 1px solid rgba(180,150,100,0.12) !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label { color: #9a8a78 !important; font-family: 'Montserrat', sans-serif !important; }

/* Chat messages */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
}

/* Message content bubbles */
[data-testid="stChatMessageContent"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(180,150,100,0.12) !important;
    border-left: 2px solid rgba(201,169,110,0.4) !important;
    border-radius: 2px !important;
    color: #d8d0c4 !important;
    font-size: 0.83rem !important;
    line-height: 1.8 !important;
    font-family: 'Montserrat', sans-serif !important;
}

/* Chat input */
[data-testid="stChatInput"] textarea {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(180,150,100,0.2) !important;
    border-radius: 2px !important;
    color: #000000 !important;
    font-family: 'Montserrat', sans-serif !important;
    font-size: 0.82rem !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: rgba(201,169,110,0.5) !important;
    box-shadow: none !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #4a3f32 !important; }
[data-testid="stChatInputSubmitButton"] svg { fill: #c9a96e !important; }
[data-testid="stBottom"] {
    background: #0a0a0a !important;
    border-top: 1px solid rgba(180,150,100,0.08) !important;
}

/* Buttons */
.stButton > button {
    background: transparent !important;
    border: 1px solid rgba(180,150,100,0.22) !important;
    border-radius: 2px !important;
    color: #7a6a58 !important;
    font-family: 'Montserrat', sans-serif !important;
    font-size: 0.62rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    transition: all 0.25s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    border-color: rgba(201,169,110,0.5) !important;
    color: #c9a96e !important;
    background: rgba(201,169,110,0.05) !important;
}

/* Spinner */
[data-testid="stSpinner"] > div { border-top-color: #c9a96e !important; }

/* Markdown in chat */
[data-testid="stChatMessageContent"] strong { color: #c9a96e !important; }
[data-testid="stChatMessageContent"] p { margin-bottom: 0.4rem !important; }
[data-testid="stChatMessageContent"] ul { padding-left: 1.2rem !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(201,169,110,0.18); border-radius: 2px; }

/* Column gaps */
[data-testid="column"] { padding: 0 0.25rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Load orchestrator ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_orchestrator():
    try:
        from orchestrator import run_orchestrator
        return run_orchestrator, None
    except Exception as e:
        return None, str(e)

run_fn, _err = load_orchestrator()

# ── Session state ─────────────────────────────────────────────────────────────
if "messages"      not in st.session_state: st.session_state.messages      = []
if "chat_history"  not in st.session_state: st.session_state.chat_history  = []
if "last_agent"    not in st.session_state: st.session_state.last_agent    = None
if "turn_count"    not in st.session_state: st.session_state.turn_count    = 0

# ── Helpers ───────────────────────────────────────────────────────────────────
def detect_agent(text: str):
    t = text.lower()
    if any(s in t for s in ["return decision", "price paid", "order #", "return window", "eligible"]):
        return "support"
    if any(s in t for s in ["recommendation", "in stock", "bestseller", "budget"]):
        return "shopper"
    return None

def chat(user_input: str) -> str:
    if run_fn is None:
        return f"⚠️ Backend failed to load: {_err}"
    try:
        import orchestrator as _orch
        _orch.chat_history = st.session_state.chat_history
        response = run_fn(user_input)
        st.session_state.chat_history = _orch.chat_history
        return response
    except Exception as e:
        return f"Error: {e}"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family:"Cormorant Garamond",serif;font-size:1.5rem;
    font-weight:300;letter-spacing:0.3em;color:#c9a96e;
    text-transform:uppercase;padding:1rem 0 0.25rem'>◆ Maison</div>
    <div style='font-size:0.58rem;letter-spacing:0.25em;text-transform:uppercase;
    color:#4a3a2a;margin-bottom:1.5rem'>Retail AI Assistant</div>
    <hr style='border:none;border-top:1px solid rgba(180,150,100,0.12);margin:0 0 1.2rem'>
    """, unsafe_allow_html=True)

    agent_color = {"shopper": "#c9a96e", "support": "#a0c0dc"}.get(
        st.session_state.last_agent, "#4a3a2a"
    )
    agent_label = {
        "shopper": "Personal Shopper",
        "support": "Customer Support"
    }.get(st.session_state.last_agent, "Awaiting query")

    st.markdown(f"""
    <div style='margin-bottom:1.2rem'>
        <div style='font-size:0.58rem;letter-spacing:0.22em;text-transform:uppercase;
        color:#4a3a2a;margin-bottom:0.5rem'>Active Agent</div>
        <div style='font-size:0.75rem;color:{agent_color};font-weight:500'>
            {'◉' if st.session_state.last_agent else '○'} {agent_label}
        </div>
    </div>
    <div style='margin-bottom:1.2rem'>
        <div style='font-size:0.58rem;letter-spacing:0.22em;text-transform:uppercase;
        color:#4a3a2a;margin-bottom:0.5rem'>Session</div>
        <div style='font-size:0.75rem;color:#7a6a58'>
            {st.session_state.turn_count} turn{'s' if st.session_state.turn_count != 1 else ''}
        </div>
    </div>
    <hr style='border:none;border-top:1px solid rgba(180,150,100,0.08);margin:0 0 1.2rem'>
    <div style='font-size:0.58rem;letter-spacing:0.22em;text-transform:uppercase;
    color:#4a3a2a;margin-bottom:0.6rem'>Capabilities</div>
    <div style='font-size:0.72rem;color:#5a4a3a;line-height:2'>
        ◆ Product Search<br>
        ◆ Style Recommendations<br>
        ◆ Order Lookup<br>
        ◆ Return Evaluation
    </div>
    <hr style='border:none;border-top:1px solid rgba(180,150,100,0.08);margin:1.2rem 0'>
    """, unsafe_allow_html=True)

    if st.button("✕  Clear Conversation"):
        st.session_state.messages     = []
        st.session_state.chat_history = []
        st.session_state.last_agent   = None
        st.session_state.turn_count   = 0
        st.rerun()

    if _err:
        st.error(f"Import error: {_err}")

# ── Main layout ───────────────────────────────────────────────────────────────
_, col, _ = st.columns([1, 3, 1])

with col:
    # Header
    st.markdown("""
    <div style='text-align:center;padding:2.5rem 0 1.8rem'>
        <div style='font-family:"Cormorant Garamond",serif;font-size:2.8rem;
        font-weight:300;letter-spacing:0.3em;color:#f0ece4;text-transform:uppercase'>
            Mai<span style='color:#c9a96e'>s</span>on
        </div>
        <div style='font-size:0.6rem;font-weight:500;letter-spacing:0.3em;
        color:#4a3a2a;text-transform:uppercase;margin-top:0.4rem'>
            Intelligent Retail Assistant
        </div>
        <div style='width:50px;height:1px;
        background:linear-gradient(90deg,transparent,#c9a96e,transparent);
        margin:1.2rem auto 0'></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Welcome + chips (only when empty) ────────────────────────────────────
    if not st.session_state.messages:
        st.markdown("""
        <div style='text-align:center;padding:1.5rem 0 1.8rem'>
            <div style='font-family:"Cormorant Garamond",serif;font-size:1.15rem;
            font-weight:300;font-style:italic;color:#4a3a2a;margin-bottom:0.35rem'>
                How may I assist you today?
            </div>
            <div style='font-size:0.58rem;letter-spacing:0.25em;
            text-transform:uppercase;color:#2a2018'>Shop · Discover · Support</div>
        </div>
        <div style='font-size:0.58rem;letter-spacing:0.22em;text-transform:uppercase;
        color:#3a2e22;text-align:center;margin-bottom:0.7rem'>Try asking</div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        chips = [
            (c1, "◆ Shop",    "Find me an evening dress under $200 in size 8"),
            (c2, "◆ Sale",    "Show me sale items for a summer wedding in size 10"),
            (c3, "◆ Returns", "I want to return order O0002, it doesn't fit"),
        ]
        for col_obj, label, prompt in chips:
            with col_obj:
                if st.button(label, key=f"chip_{label}"):
                    st.session_state._chip = prompt
                    st.rerun()

    # ── Chat history ──────────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        role = "user" if msg["role"] == "user" else "assistant"
        with st.chat_message(role):
            st.markdown(msg["content"])

    # ── Input ─────────────────────────────────────────────────────────────────
    user_input = st.chat_input("Ask about products, orders, or returns…")

    if "_chip" in st.session_state:
        user_input = st.session_state._chip
        del st.session_state._chip

    if user_input and user_input.strip():
        query = user_input.strip()

        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.turn_count += 1

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner(""):
                response = chat(query)
            st.markdown(response)

        agent = detect_agent(response)
        if agent:
            st.session_state.last_agent = agent

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
