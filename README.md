# Maison — Retail AI Assistant

An intelligent AI assistant for a premium fashion retail store, built with LangGraph and Groq. One agent handles product discovery, style recommendations, order inquiries, and return evaluations.

---

## Features

- **Product Search** — filter by size, price, tags, sale/clearance status
- **Style Recommendations** — ranked by stock availability, bestseller score, and budget
- **Order Lookup** — fetch order details with linked product info
- **Return Evaluation** — policy-based return eligibility decisions

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq — `llama-3.3-70b-versatile` |
| Agent Framework | LangGraph |
| Database | Supabase (PostgreSQL) |
| UI | Streamlit |
| Retries | Tenacity |

---

## Project Structure

```
Retail_AI_Assistant/
├── app.py               # Streamlit UI
├── agent.py             # Single agent — all tools and graph
├── supabase_client.py   # Shared Supabase client
├── requirements.txt
└── .env
```

---

## Setup

**1. Clone the repository**
```bash
git clone https://github.com/Subhhankar/Retail_AI_Assistant
cd Retail_AI_Assistant
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Configure environment variables**

Create a `.env` file in the root directory:
```env
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
GROQ_API_KEY=your_groq_api_key
```

**4. Enable Supabase Row Level Security**

Run this in your Supabase SQL editor:
```sql
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;
CREATE POLICY "allow_all" ON orders FOR SELECT USING (true);

ALTER TABLE product_inventory ENABLE ROW LEVEL SECURITY;
CREATE POLICY "allow_all" ON product_inventory FOR SELECT USING (true);
```

**5. Run the app**
```bash
streamlit run app.py
```

---

---

## Database Schema

**`product_inventory`**
`product_id`, `title`, `vendor`, `price`, `compare_at_price`, `tags`, `sizes_available`, `stock_per_size`, `is_sale`, `is_clearance`, `bestseller_score`

**`orders`**
`order_id`, `order_date`, `size`, `price_paid`, `customer_id`, `product_id` (FK → product_inventory)
