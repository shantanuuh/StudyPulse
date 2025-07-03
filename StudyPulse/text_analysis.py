import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np
from transformers import pipeline
import nltk
from student_responses_analysis import analyze_student_responses
from functools import lru_cache
import time
import logging

# ✅ Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ✅ Download NLTK resources at runtime
nltk.data.path.append("/home/appuser/nltk_data")  # Ensure correct path for Streamlit Cloud
nltk.download('punkt')
nltk.download('stopwords')

# ✅ Cache the Sentiment Analyzer Model
@st.cache_resource
@lru_cache(maxsize=1)
def load_sentiment_analyzer():
    """Loads and caches the sentiment analysis model."""
    try:
        with st.spinner("Loading sentiment analysis model..."):
            return pipeline("sentiment-analysis")
    except Exception as e:
        st.error(f"❌ Failed to load sentiment analysis model: {str(e)}")
        return None

# ✅ Load sentiment model
sentiment_analyzer = load_sentiment_analyzer()
if sentiment_analyzer is None:
    st.error("❌ Sentiment analysis model could not be loaded. Please check your transformers installation.")

def text_analysis_page():
    """Render text analysis page with NLP insights."""
    st.markdown("<h2 class='sub-header'>📊 Text Response Analysis</h2>", unsafe_allow_html=True)

    if st.session_state.get("data") is None:
        st.warning("⚠️ Please upload data first on the Home page!")
        return

    df = st.session_state.data

    potential_text_cols = [
        col for col in df.columns
        if df[col].dtype == "object" and df[col].dropna().astype(str).str.split().str.len().mean() > 3
    ]

    if not potential_text_cols:
        st.warning("⚠️ No suitable text columns found for analysis. Please upload a dataset with open-ended responses.")
        return

    selected_columns = st.multiselect(
        "📌 Select text columns to analyze:",
        options=potential_text_cols,
        default=st.session_state.get("selected_columns", potential_text_cols[:min(3, len(potential_text_cols))])
    )

    if not selected_columns:
        st.info("ℹ️ Please select at least one text column to analyze.")
        return

    with st.expander("⚙️ Analysis Options"):
        sentiment_option = st.checkbox("Include Sentiment Analysis", value=True)
        keyword_option = st.checkbox("Include Keyword Extraction", value=True)
        pattern_option = st.checkbox("Include Pattern Analysis", value=True)

    progress_bar = st.progress(0)

    if st.button("🔍 Analyze Selected Text Columns"):
        st.session_state.text_analysis_results = {}

        with st.spinner("🔄 Analyzing text responses..."):
            progress_bar = st.progress(0)

            for i, col in enumerate(selected_columns):
                progress_bar.progress((i + 1) / len(selected_columns))
                st.write(f"📊 Analyzing: {col}")

                responses = df[col].dropna().astype(str).tolist()
                tuple_responses = tuple(responses)

                if sentiment_option:
                    analysis_result = analyze_student_responses(tuple_responses, col, sentiment_analyzer)
                else:
                    analysis_result = analyze_student_responses(tuple_responses, col)

                st.session_state.text_analysis_results[col] = analysis_result

        progress_bar.progress(100)
        time.sleep(0.5)
        progress_bar.empty()

        st.success("✅ Analysis Complete!")

    if st.session_state.get("text_analysis_results"):
        tabs = st.tabs([col if len(col) <= 20 else col[:20] + "..." for col in st.session_state.text_analysis_results.keys()])

        for i, (col, analysis) in enumerate(st.session_state.text_analysis_results.items()):
            with tabs[i]:
                st.subheader(f"📌 Analysis of: {col}")

                if "error" in analysis:
                    st.error(analysis["error"])
                    continue

                col1, col2 = st.columns([3, 2])

                # ✅ Keyword Analysis
                if keyword_option and "keyword_analysis" in analysis:
                    with col1:
                        st.write("### 🔑 Keyword Analysis")
                        if "error" in analysis["keyword_analysis"]:
                            st.warning(analysis["keyword_analysis"]["error"])
                        else:
                            top_words = analysis["keyword_analysis"]["top_words"]
                            if top_words:
                                fig = px.bar(
                                    x=[word[0] for word in top_words],
                                    y=[word[1] for word in top_words],
                                    title="Top Keywords",
                                    labels={"x": "Words", "y": "Frequency"},
                                    color_discrete_sequence=px.colors.qualitative.Set3
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("No significant keywords found.")

                # ✅ Sentiment Analysis
                if sentiment_option and "sentiment_analysis" in analysis:
                    with col2:
                        st.write("### 📈 Sentiment Analysis")
                        if "error" in analysis["sentiment_analysis"]:
                            st.warning(analysis["sentiment_analysis"]["error"])
                        else:
                            sentiment_counts = analysis["sentiment_analysis"]["counts"]
                            fig = px.pie(
                                values=list(sentiment_counts.values()),
                                names=list(sentiment_counts.keys()),
                                title="Sentiment Distribution",
                                color_discrete_sequence=px.colors.sequential.Blugrn
                            )
                            st.plotly_chart(fig, use_container_width=True)

                # ✅ Pattern Analysis
                if pattern_option and "pattern_analysis" in analysis and "common_patterns" in analysis["pattern_analysis"]:
                    st.write("### 🧩 Pattern Clusters")
                    patterns = analysis["pattern_analysis"]["common_patterns"]
                    for pattern, details in patterns.items():
                        st.markdown(f"**{pattern}** ({details['percentage']:.1f}% of responses)")
                        st.write(f"**Representative Response:** {details['representative_response']}")
                        st.write(f"**Top Keywords:** {', '.join(details['keywords'])}")
                        st.write("---")

    if st.session_state.get("text_analysis_results") and st.button("📥 Export Text Analysis Results"):
        report = "# 📊 Text Analysis Report\n\n"
        for col, analysis in st.session_state.text_analysis_results.items():
            report += f"## 📌 {col}\n\n"
            if "sentiment_analysis" in analysis:
                report += "### 📈 Sentiment Analysis\n\n"
                for sentiment, percentage in analysis["sentiment_analysis"]["counts"].items():
                    report += f"- **{sentiment}:** {percentage:.1f}%\n"
            if "keyword_analysis" in analysis:
                report += "\n### 🔑 Top Keywords\n\n"
                for word, count in analysis["keyword_analysis"]["top_words"]:
                    report += f"- {word}: {count}\n"
            if "pattern_analysis" in analysis and "common_patterns" in analysis["pattern_analysis"]:
                report += "\n### 🧩 Patterns\n\n"
                for pattern, details in analysis["pattern_analysis"]["common_patterns"].items():
                    report += f"**{pattern}** ({details['percentage']:.1f}%)\n\n"
                    report += f"- **Representative Response:** {details['representative_response']}\n"
                    report += f"- **Top Keywords:** {', '.join(details['keywords'])}\n\n"
            report += "---\n\n"

        st.download_button(
            label="📥 Download Report",
            data=report,
            file_name="text_analysis_report.md",
            mime="text/markdown"
        )
