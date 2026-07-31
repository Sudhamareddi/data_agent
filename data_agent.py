"""
Data/Analytics Agent (Free Version — uses Groq)
-------------------------------------------------
Takes a natural-language question (e.g. "what were the top 3 products by
revenue in the North region?"), decides to query the sales database using
tool-calling, runs the SQL, and returns a plain-English answer.

Uses Groq's free tier (fast open models like Llama 3.3, no cost) instead of
a paid API, so this whole project can be built and hosted for free.

HOW IT WORKS
------------
1. We give the model a "run_sql_query" tool and describe the database schema.
2. The model decides what SQL to write to answer the question, and calls the tool.
3. We execute that SQL against sales.db and return the results to the model.
4. The model turns the raw rows into a clear, human-readable answer.

SETUP (all free)
-----------------
1. pip install groq --break-system-packages
2. Run setup_db.py first to create sales.db
3. Get a free API key at https://console.groq.com (sign up, no card needed
   as of writing — always double check current terms)
4. Set it: export GROQ_API_KEY="your-key-here"
5. python data_agent.py
"""

import sqlite3
import json
import os
from groq import Groq

client = Groq()  # reads GROQ_API_KEY from environment
MODEL = "llama-3.3-70b-versatile"  # fast, free-tier model with tool-calling support

DB_PATH = "sales.db"

# ---- Describe the schema so the model knows what it can query ----
SCHEMA_DESCRIPTION = """
Table: sales
Columns:
  id INTEGER
  order_date TEXT (format: YYYY-MM-DD)
  product TEXT
  category TEXT (Electronics, Furniture, Stationery)
  region TEXT (North, South, East, West)
  quantity INTEGER
  unit_price REAL
  revenue REAL (quantity * unit_price)
"""

# ---- Tool definition (OpenAI-style function schema, which Groq uses) ----
tools = [
    {
        "type": "function",
        "function": {
            "name": "run_sql_query",
            "description": (
                "Run a read-only SQL SELECT query against the sales database "
                "and return the results. Use this to answer any question about "
                "sales, revenue, products, regions, or categories."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A valid SQLite SELECT query.",
                    }
                },
                "required": ["query"],
            },
        },
    }
]


def run_sql_query(query: str):
    """Executes a SQL query against sales.db and returns rows as a list of dicts."""
    # Basic safety check: only allow SELECT statements
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


def ask_agent(question: str) -> str:
    """
    Sends the user's question to the model, lets it call the SQL tool as
    needed, and returns the final natural-language answer.
    """
    messages = [
        {
            "role": "system",
            "content": (
                f"You are a data analytics assistant. Here is the database schema:\n"
                f"{SCHEMA_DESCRIPTION}\n"
                f"Use the run_sql_query tool to answer the user's questions."
            ),
        },
        {"role": "user", "content": question},
    ]

    # Loop until the model stops calling tools and gives a final answer
    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
        )

        msg = response.choices[0].message

        if not msg.tool_calls:
            # No more tool calls -> this is the final answer
            return msg.content

        # Add the assistant's tool-call message to history
        messages.append(msg)

        # Execute each requested tool call and send results back
        for tool_call in msg.tool_calls:
            if tool_call.function.name == "run_sql_query":
                args = json.loads(tool_call.function.arguments)
                result = run_sql_query(args["query"])
            else:
                result = {"error": f"Unknown tool: {tool_call.function.name}"}

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            })


if __name__ == "__main__":
    print("Data Analytics Agent — ask a question about the sales data (or 'quit' to exit)\n")
    while True:
        question = input("You: ")
        if question.strip().lower() in ("quit", "exit"):
            break
        answer = ask_agent(question)
        print(f"\nAgent: {answer}\n")
