"""
Data/Analytics Agent — v3 (agent discovers schema itself)
--------------------------------------------------------------
Upgrade from v2: previously we handed the full schema to the agent
upfront in the system prompt. Now the agent is given ONLY 3 tools and
has to explore the database on its own, the way a real analyst would
approach an unfamiliar database:

  1. list_tables        -> "what tables exist?"
  2. get_table_schema    -> "what columns does this table have?"
  3. run_sql_query       -> "now run this SQL"

This means the agent's first move on a brand-new question is usually to
call list_tables and get_table_schema BEFORE it writes any SQL - you can
literally watch it explore the database step by step.

SETUP
-----
1. pip install groq --break-system-packages
2. Have chinook.db in the same folder (see setup_chinook.py)
3. export GROQ_API_KEY="your-key-here"
4. python data_agent.py
"""

import sqlite3
import json
from groq import Groq

client = Groq()
MODEL = "llama-3.3-70b-versatile"

DB_PATH = "chinook.db"

# ---- Tool 1: list all tables (no columns yet — agent must ask separately) ----
def list_tables():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    conn.close()
    return {"tables": tables}


# ---- Tool 2: inspect one table's columns ----
def get_table_schema(table_name: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table_name})")
    cols = cur.fetchall()
    conn.close()
    if not cols:
        return {"error": f"Table '{table_name}' not found."}
    return {
        "table": table_name,
        "columns": [{"name": c[1], "type": c[2]} for c in cols],
    }


# ---- Tool 3: run the actual query ----
def run_sql_query(query: str):
    if not query.strip().lower().startswith("select"):
        return {"error": "Only SELECT queries are allowed."}
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        cur.execute(query)
        rows = [dict(row) for row in cur.fetchall()]
        return {"rows": rows}
    except Exception as e:
        return {"error": str(e)}
    finally:
        conn.close()


tools = [
    {
        "type": "function",
        "function": {
            "name": "list_tables",
            "description": "List all table names available in the database. Call this first if you don't yet know what tables exist.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_table_schema",
            "description": "Get the column names and types for a specific table. Call this before writing SQL that uses a table you haven't inspected yet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {"type": "string", "description": "Exact table name, e.g. 'Customer'"}
                },
                "required": ["table_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_sql_query",
            "description": "Run a read-only SQL SELECT query (joins allowed) once you know the relevant table names and columns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "A valid SQLite SELECT query."}
                },
                "required": ["query"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "list_tables": lambda args: list_tables(),
    "get_table_schema": lambda args: get_table_schema(args["table_name"]),
    "run_sql_query": lambda args: run_sql_query(args["query"]),
}


def ask_agent(question: str, history: list | None = None):
    """
    Sends the question to the model. The model must explore the database
    itself (list_tables / get_table_schema) before it can run SQL, unless
    it already learned the structure earlier in this conversation.

    Returns a dict:
      {
        "answer": final natural-language answer (str),
        "sql": the last SQL query actually run (str or None),
        "rows": the raw rows from that query (list or None),
        "tool_log": ordered list of every tool call made this turn,
                    e.g. ["list_tables", "get_table_schema(Customer)", "run_sql_query(...)"]
        "history": updated message history to pass into the next call
      }
    """
    if history is None:
        history = [
            {
                "role": "system",
                "content": (
                    "You are a data analytics assistant for a music store database. "
                    "You do NOT know the schema in advance. Before writing any SQL, "
                    "use list_tables and get_table_schema to discover what tables and "
                    "columns exist. Only call run_sql_query once you know the correct "
                    "table and column names. Remember what you've already learned "
                    "earlier in the conversation — no need to re-discover the same "
                    "table twice."
                ),
            }
        ]

    messages = history + [{"role": "user", "content": question}]

    last_sql = None
    last_rows = None
    tool_log = []

    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content})
            return {
                "answer": msg.content,
                "sql": last_sql,
                "rows": last_rows,
                "tool_log": tool_log,
                "history": messages,
            }

        messages.append(msg)

        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments or "{}")

            if name in TOOL_FUNCTIONS:
                result = TOOL_FUNCTIONS[name](args)
            else:
                result = {"error": f"Unknown tool: {name}"}

            # Log what the agent did, in plain form, for display in the UI
            if name == "list_tables":
                tool_log.append("Explored: listed all tables")
            elif name == "get_table_schema":
                tool_log.append(f"Explored: inspected columns of '{args.get('table_name')}'")
            elif name == "run_sql_query":
                last_sql = args.get("query")
                last_rows = result.get("rows")
                tool_log.append(f"Ran SQL query")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            })


if __name__ == "__main__":
    print("Data Analytics Agent (self-discovering schema) — 'quit' to exit\n")
    conversation_history = None
    while True:
        question = input("You: ")
        if question.strip().lower() in ("quit", "exit"):
            break
        result = ask_agent(question, conversation_history)
        conversation_history = result["history"]

        print("\n--- Agent's exploration steps ---")
        for step in result["tool_log"]:
            print(f"  • {step}")
        if result["sql"]:
            print(f"\n[Final SQL used: {result['sql']}]")
        print(f"\nAgent: {result['answer']}\n")
