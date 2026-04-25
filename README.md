# Maison — AI Retail Assistant

A multi-agent AI assistant for fashion retail, built with LangGraph, Groq (LLaMA 3.3), Supabase, and Streamlit.

## Agents

| Agent | Tools | Handles |
|---|---|---|
| Personal Shopper | `search_products`, `get_product` | Browse, search, product recommendations |
| Customer Support | `get_order`, `evaluate_return` | Order lookup, return eligibility |

An **orchestrator** classifies every incoming message and routes it to the right agent automatically.

## Project Structure

```
├── app.py                  # Streamlit UI
├── orchestrator.py         # LLM-based intent router
├── supabase_client.py      # Shared Supabase client
├── agents/
│   ├── __init__.py
│   ├── shopper_agent.py    # Personal shopper graph + runner
│   └── support_agent.py    # Customer support graph + runner
├── requirements.txt
└── .env.example
```

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-username/maison-ai.git
cd maison-ai
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:

```
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_service_role_key
GROQ_API_KEY=your_groq_api_key
```

### 5. Set up Supabase RLS

Run this in the Supabase SQL editor to allow reads:

```sql
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;
CREATE POLICY "allow_all" ON orders FOR SELECT USING (true);

ALTER TABLE product_inventory ENABLE ROW LEVEL SECURITY;
CREATE POLICY "allow_all" ON product_inventory FOR SELECT USING (true);
```

### 6. Run the app

```bash
streamlit run app.py
```

## Return Policy

| Condition | Outcome |
|---|---|
| Clearance item | ❌ Not eligible — final sale |
| Order > 30 days old | ❌ Return window expired |
| Sale item within 30 days | ✅ Store credit only |
| Regular item within 30 days | ✅ Full refund |

## Tech Stack

- **LLM** — Groq `llama-3.3-70b-versatile`
- **Agent framework** — LangGraph
- **Database** — Supabase (PostgreSQL)
- **UI** — Streamlit
- **Retries** — Tenacity