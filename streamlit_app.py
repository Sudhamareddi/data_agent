"""
Streamlit chat UI for the Data Analytics Agent.

Run locally: streamlit run streamlit_app.py
(Make sure setup_db.py has been run first, and GROQ_API_KEY is set.)

To deploy for FREE on Streamlit Community Cloud:
  1. Push this repo to GitHub (must include: streamlit_app.py, data_agent.py,
     sales.db, requirements.txt)
  2. Go to share.streamlit.io, sign in with GitHub (free, no card needed)
  3. Click "New app", select your repo and this file (streamlit_app.py)
  4. In "Advanced settings" -> Secrets, add:
        GROQ_API_KEY = "your-key-here"
  5. Deploy -> you get a public URL like:
     https://your-app-name.streamlit.app
"""

import streamlit as st
from data_agent import ask_agent

st.set_page_config(page_title="Sales Data Analytics Agent", page_icon="📊")

st.title("📊 Sales Data Analytics Agent")
st.write(
    "Ask natural-language questions about a sample sales database "
    "(products, regions, revenue, categories). The agent writes and runs "
    "SQL queries behind the scenes to answer you."
)

with st.expander("Example questions"):
    st.write("- What were the top 5 products by total revenue?")
    st.write("- Which region had the highest total sales?")
    st.write("- What is the average order value in the Furniture category?")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input("Ask a question about the sales data...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ask_agent(question)
        st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
