import streamlit as st

st.set_page_config(
    page_title="Movie Success AI",
    layout="centered"
)

st.title("🎬 Movie Success Prediction System")
st.write("Production-grade ML system using IMDb data")

st.markdown("Use the pages from the sidebar:")
st.markdown("- **Prediction** → Decision")
st.markdown("- **What-If Simulation** → Sensitivity analysis")
