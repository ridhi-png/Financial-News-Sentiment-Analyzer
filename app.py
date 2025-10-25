
import streamlit as st

APP_LINK = "https://unreverberated-melodie-prevailingly.ngrok-free.dev/#probabilities"

st.set_page_config(page_title="Financial News Sentiment Analyzer", page_icon="💹", layout="centered")

st.title("💹 Financial News Sentiment Analyzer App")
st.write("Analyze financial headlines and get sentiment predictions powered by FinBERT (ProsusAI/finbert).")

st.markdown(
    f"""
    🌐 **Live App:** [👉 Open Hosted Version]({https://unreverberated-melodie-prevailingly.ngrok-free.dev/#probabilities})
    """,
    unsafe_allow_html=True
)

st.divider()  # Adds a nice horizontal separator


