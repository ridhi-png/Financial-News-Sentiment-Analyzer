# app.py
import streamlit as st
import pandas as pd
from sentiment_model import analyze_sentiment

st.set_page_config(page_title="Financial News Sentiment Analyzer", page_icon="🧠", layout="centered")

st.title(" Financial News Sentiment Analyzer")
st.write("Analyze finance-related headlines and news using AI (FinBERT).")

# --- Input ---
headline = st.text_area("Enter a financial headline or short news snippet:",
                        "Apple shares rose after strong Q4 earnings report")

if st.button(" Analyze Sentiment"):
    sentiment, result = analyze_sentiment(headline)
    st.subheader(f"Sentiment: **{sentiment.upper()}**")
    st.write(result)

# --- Upload CSV ---
st.markdown("---")
st.header(" Analyze Multiple Headlines (CSV Upload)")
uploaded_file = st.file_uploader("Upload a CSV file with a 'headline' column")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    sentiments = []
    for text in df['headline']:
        s, _ = analyze_sentiment(text)
        sentiments.append(s)
    df['Sentiment'] = sentiments
    st.write(df)
    st.download_button("⬇️ Download Results", df.to_csv(index=False), "results.csv", "text/csv")
