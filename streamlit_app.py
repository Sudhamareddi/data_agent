"""
Streamlit chat UI for the Data Analytics Agent — v3
-----------------------------------------------------
Now shows the agent's own exploration steps (which tables/columns it
looked up before writing SQL), not just the final answer. This is the
most interesting part to demo — you can literally watch it figure out
the database structure on its own.

Run locally: streamlit run streamlit_app.py
(Make sure chinook.db is present and GROQ_API_KEY is set.)

Deploy for FREE on Streamlit Community Cloud:
  1. Push to GitHub: streamlit_app.py, data_agent.py, chinook.db, requirements.txt
  2. share.streamlit.io -> New app -> select repo -> main file: streamlit_app.py
  3. Advanced settings -> Secrets -> GROQ_API_KEY = "your-key-here"
  4. Deploy
"""

import streamlit as st
import pandas as pd
from data_agent import ask_agent

st.set_page_config(page_title="Music Store Data Agent", page_icon="🎵", layout="centered")

with st.sidebar:
    st.subheader("Session")
    st.write(
        "This agent remembers earlier questions in the conversation. If it "
        "seems confused or stuck on a bad guess from before, reset here."
    )
    if st.button("🔄 New conversation (clear memory)"):
        st.session_state.display_messages = []
        st.session_state.agent_history = None
        st.rerun()

st.title("🎵 Music Store Data Analytics Agent")
st.write(
    "Ask natural-language questions about a real multi-table music store "
    "database. The agent doesn't know the schema in advance — watch it "
    "explore the tables and columns itself before writing SQL."
)

with st.expander("Example questions"):
    st.write("- Who are the top 5 customers by total spending?")
    st.write("- Which genre has the most tracks?")
    st.write("- Which employee has generated the most sales?")
    st.write("- Now break that down by country")

if "display_messages" not in st.session_state:
    st.session_state.display_messages = []
if "agent_history" not in st.session_state:
    st.session_state.agent_history = None


def try_build_chart(rows):
    if not rows or len(rows) < 2:
        return None
    df = pd.DataFrame(rows)
    if df.shape[1] != 2:
        return None
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) != 1:
        return None
    label_col = [c for c in df.columns if c not in numeric_cols][0]
    return df.set_index(label_col)


def render_assistant_extras(sql, tool_log, chart_df, rows):
    if tool_log:
        with st.expander(f"🔍 Agent's exploration steps ({len(tool_log)})"):
            for step in tool_log:
                st.write(f"- {step}")
    if sql:
        with st.expander("SQL query used"):
            st.code(sql, language="sql")
    # Always show the actual raw result, regardless of what the agent's
    # text summary says — this is the ground truth, so if the summary text
    # is ever wrong, you can still see the real data right here.
    if rows:
        with st.expander(f"Raw query result ({len(rows)} row{'s' if len(rows) != 1 else ''})"):
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
    if chart_df is not None:
        st.bar_chart(chart_df)


for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant":
            render_assistant_extras(
                msg.get("sql"), msg.get("tool_log"), msg.get("chart"), msg.get("rows")
            )

question = st.chat_input("Ask a question about the music store data...")

if question:
    st.session_state.display_messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Exploring the database and thinking..."):
            result = ask_agent(question, st.session_state.agent_history)
        st.session_state.agent_history = result["history"]

        st.write(result["answer"])

        chart_df = try_build_chart(result["rows"]) if result["rows"] else None
        render_assistant_extras(result["sql"], result["tool_log"], chart_df, result["rows"])

    st.session_state.display_messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sql": result["sql"],
        "tool_log": result["tool_log"],
        "chart": chart_df,
        "rows": result["rows"],
    })
