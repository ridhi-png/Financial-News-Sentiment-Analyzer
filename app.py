import streamlit as st
from transformers import pipeline
import plotly.graph_objects as go

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Financial News Sentiment Analyzer",
    page_icon="📈",
    layout="centered"
)

# ── Load model (cached so it only loads once) ─────────────────────────────────
@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        return_all_scores=True
    )

classifier = load_model()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📈 Financial News Sentiment Analyzer")
st.write("Analyze financial headlines and get sentiment predictions powered by **FinBERT** (ProsusAI/finbert).")
st.divider()

# ── Input ─────────────────────────────────────────────────────────────────────
st.subheader("Enter a financial headline")

headline = st.text_area(
    label="Headline",
    placeholder="e.g. Apple reports record quarterly revenue, beating analyst expectations by 12%",
    height=100,
    label_visibility="collapsed"
)

analyze_btn = st.button("Analyze Sentiment", type="primary", use_container_width=True)

# ── Sample headlines ──────────────────────────────────────────────────────────
st.markdown("**Or try a sample:**")
col1, col2, col3 = st.columns(3)

sample_positive = "Apple beats earnings expectations with record iPhone sales"
sample_negative = "Startup lays off 40% of staff amid funding crisis"
sample_neutral  = "Federal Reserve holds interest rates steady at 5.25%"

if col1.button("📗 Positive example", use_container_width=True):
    headline = sample_positive
    analyze_btn = True

if col2.button("📕 Negative example", use_container_width=True):
    headline = sample_negative
    analyze_btn = True

if col3.button("📘 Neutral example", use_container_width=True):
    headline = sample_neutral
    analyze_btn = True

# ── Analysis ──────────────────────────────────────────────────────────────────
if analyze_btn and headline.strip():
    with st.spinner("Analyzing..."):
        results = classifier(headline.strip())[0]

    # Map results to a clean dict
    scores = {r["label"].capitalize(): round(r["score"] * 100, 2) for r in results}
    top_label = max(scores, key=scores.get)
    top_score = scores[top_label]

    st.divider()

    # ── Verdict ───────────────────────────────────────────────────────────────
    color_map = {"Positive": "🟢", "Negative": "🔴", "Neutral": "🟡"}
    st.subheader(f"{color_map.get(top_label, '⚪')} Verdict: **{top_label}**")
    st.caption(f"Model confidence: {top_score}%")

    # ── Probability bar chart ─────────────────────────────────────────────────
    labels = list(scores.keys())
    values = list(scores.values())
    bar_colors = []
    for l in labels:
        if l == "Positive":
            bar_colors.append("#2ecc71")
        elif l == "Negative":
            bar_colors.append("#e74c3c")
        else:
            bar_colors.append("#f1c40f")

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=bar_colors,
        text=[f"{v}%" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        yaxis=dict(range=[0, 110], title="Confidence (%)"),
        xaxis_title="Sentiment",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20, b=20),
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Business explanation ──────────────────────────────────────────────────
    st.divider()
    st.subheader("What this means")

    explanations = {
        "Positive": (
            f"The headline carries a **positive** signal with {top_score}% confidence. "
            "This typically suggests favourable news — strong earnings, growth, or market optimism. "
            "Investors may interpret this as a potential buy signal or confirmation of bullish sentiment."
        ),
        "Negative": (
            f"The headline carries a **negative** signal with {top_score}% confidence. "
            "This may indicate bad news — losses, layoffs, regulatory issues, or market pessimism. "
            "Analysts might treat this as a risk flag or bearish indicator."
        ),
        "Neutral": (
            f"The headline is **neutral** with {top_score}% confidence. "
            "This typically reflects factual reporting without strong directional bias — "
            "policy announcements, routine disclosures, or mixed signals that don't clearly lean either way."
        ),
    }
    st.markdown(explanations.get(top_label, ""))

    # ── Raw scores table ──────────────────────────────────────────────────────
    with st.expander("See all probability scores"):
        for label, score in sorted(scores.items(), key=lambda x: -x[1]):
            st.markdown(f"**{label}:** {score}%")
            st.progress(score / 100)

elif analyze_btn and not headline.strip():
    st.warning("Please enter a headline first.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Built by Ridhi Arora · IIT Jodhpur · Powered by ProsusAI/FinBERT via HuggingFace Transformers")
