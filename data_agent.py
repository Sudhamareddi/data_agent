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
MODEL = "openai/gpt-oss-120b"  # stronger tool-calling reliability than llama-3.3-70b for structured multi-step tasks

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
                    "table twice.\n\n"
                    "IMPORTANT: Never guess a table or column name. If a table you "
                    "expect doesn't appear in list_tables' results, look through the "
                    "actual table list for the closest real match instead of assuming "
                    "a name like 'Sales' exists — for example, sales/revenue data may "
                    "be stored under a differently-named table (e.g. Invoice), and "
                    "relationships between tables may go through a foreign key column "
                    "rather than a direct table name match. Always inspect a table's "
                    "schema with get_table_schema before assuming its columns.\n\n"
                    "CRITICAL: When you write SQL, use table and column names EXACTLY "
                    "as they were returned by list_tables and get_table_schema — same "
                    "spelling, same singular/plural form, same capitalization. Never "
                    "pluralize, rename, or paraphrase a name (e.g. if the real table is "
                    "'Employee', do not write 'Employees'; if a column is 'SupportRepId', "
                    "do not invent 'EmployeeID').\n\n"
                    "Always finish by giving a direct, plain-language answer to the "
                    "user's question — never end a turn with only a tool call and no "
                    "explanation. If you are not able to call a tool, do not write out "
                    "fake tool-call syntax as text — just answer directly in plain "
                    "language using whatever you've already learned."
                ),
            }
        ]

    messages = history + [{"role": "user", "content": question}]

    last_sql = None
    last_rows = None
    tool_log = []
    seen_calls = {}  # tracks (tool_name, args_json) -> True, to catch repeated calls
    MAX_TOOL_ROUNDS = 6  # safety cap so a confused agent can't loop forever

    try:
        for round_num in range(MAX_TOOL_ROUNDS + 1):
            # On the final allowed round, force a text-only answer using
            # tool_choice="none" instead of removing `tools` entirely — Groq
            # (like OpenAI) requires `tools` to stay present if earlier messages
            # in the conversation already contain tool calls, otherwise it
            # rejects the request as invalid.
            force_final = round_num == MAX_TOOL_ROUNDS
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools,
                tool_choice="none" if force_final else "auto",
            )
            msg = response.choices[0].message

            if not msg.tool_calls:
                answer_text = (msg.content or "").strip()
                # Some models, when denied tool access, write out fake
                # tool-call syntax as plain text instead of a real answer
                # (e.g. "<function=run_sql_query>{...}</function>"). Catch
                # that so it's never shown to the user as if it were valid.
                looks_fake = "<function=" in answer_text or "<function_call" in answer_text
                if not answer_text or looks_fake:
                    # The model returned nothing useful — ask it explicitly to
                    # summarize based on whatever was found, instead of showing
                    # a blank response to the user.
                    messages.append({"role": "assistant", "content": msg.content or ""})
                    messages.append({
                        "role": "user",
                        "content": (
                            "That wasn't a valid answer (either empty, or it contained "
                            "tool-call syntax instead of a real response). Based on "
                            "everything you've already explored and queried in this "
                            "conversation, write your best direct answer now in plain "
                            "language only — no code, no tool syntax."
                        ),
                    })
                    retry = client.chat.completions.create(
                        model=MODEL, messages=messages, tools=tools, tool_choice="none"
                    )
                    retry_text = (retry.choices[0].message.content or "").strip()
                    if not retry_text or "<function=" in retry_text or "<function_call" in retry_text:
                        answer_text = (
                            "I wasn't able to find a clear answer — could you rephrase "
                            "the question, or try a simpler one?"
                        )
                    else:
                        answer_text = retry_text
                    messages.append({"role": "assistant", "content": answer_text})
                else:
                    messages.append({"role": "assistant", "content": answer_text})

                return {
                    "answer": answer_text,
                    "sql": last_sql,
                    "rows": last_rows,
                    "tool_log": tool_log,
                    "history": messages,
                }

            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            })

            for tool_call in msg.tool_calls:
                name = tool_call.function.name
                raw_args = tool_call.function.arguments or "{}"

                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    args = {}

                # Normalize the key so equivalent calls (e.g. "{}" vs "")
                # are reliably recognized as duplicates.
                call_key = (name, json.dumps(args, sort_keys=True))

                if call_key in seen_calls:
                    # The agent is repeating an identical call — don't waste
                    # a round re-running it. Show it the cached result again
                    # so it has something concrete to act on, rather than
                    # just being told "don't repeat" with nothing to work with.
                    result = {
                        "note": "You already called this earlier — reusing that result, see below. Move on to your next step.",
                        "cached_result": seen_calls[call_key],
                    }
                    tool_log.append(f"(reused earlier result for repeated call: {name})")
                else:
                    if name in TOOL_FUNCTIONS:
                        result = TOOL_FUNCTIONS[name](args)
                    else:
                        result = {"error": f"Unknown tool: {name}"}
                        tool_log.append(f"(agent tried unknown tool: {name})")

                    seen_calls[call_key] = result

                    if name == "list_tables":
                        tool_log.append("Explored: listed all tables")
                    elif name == "get_table_schema":
                        tool_log.append(f"Explored: inspected columns of '{args.get('table_name')}'")
                    elif name == "run_sql_query":
                        last_sql = args.get("query")
                        last_rows = result.get("rows")
                        tool_log.append("Ran SQL query")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                })

        # Exhausted all rounds without a clean return (shouldn't normally
        # happen since the last round forces tool_choice="none").
        return {
            "answer": "I got stuck exploring the database and couldn't finish answering — could you try rephrasing your question?",
            "sql": last_sql,
            "rows": last_rows,
            "tool_log": tool_log,
            "history": messages,
        }

    except Exception as e:
        # Surface the real error in the chat instead of failing silently
        # or crashing the whole app.
        return {
            "answer": f"⚠️ The agent hit an internal error and couldn't finish: {type(e).__name__}: {e}",
            "sql": last_sql,
            "rows": last_rows,
            "tool_log": tool_log,
            "history": history,  # roll back to last known-good history
        }


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
