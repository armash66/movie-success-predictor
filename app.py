import streamlit as st

st.set_page_config(
    page_title="Movie Success AI",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Success AI Platform")

st.markdown("""
### Welcome

This system predicts **movie success probability** using real IMDb data
and explains **why** the prediction looks the way it does.

### What this app does:
- 🔮 Predicts movie success
- 🧠 Explains contributing factors
- 🔄 Simulates improvements in real time
- 📊 Uses multiple ML models internally

➡️ Use the **left sidebar** to navigate.
""")

st.success("Start with **Prediction** from the sidebar.")
