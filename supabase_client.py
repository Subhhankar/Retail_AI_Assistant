"""
supabase_client.py
──────────────────
Shared Supabase client — imported by both agents.
Initialised once at module level so all agents share the same connection.
"""

import os
import json
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL: str = os.getenv("SUPABASE_URL")
SUPABASE_KEY: str = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise EnvironmentError(
        "Missing SUPABASE_URL or SUPABASE_KEY in environment / .env file."
    )

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def safe_tool_response(data) -> str:
    """Always returns a non-empty string — Groq requires tool content to be a non-empty string."""
    if not data:
        return "No results found."
    return json.dumps(data, default=str)