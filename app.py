import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pandas.api.types as pat
import os
from pathlib import Path

from analyzer import get_full_summary, get_llm_summary
from llm_insights import get_ai_insights
from qa_engine import ask_question, get_suggested_questions

# Page config 
st.set_page_config(
    page_title="AI Data Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load API key from .env file
def load_env_key():
    """Reads GROQ_API_KEY from .env file in the same folder as app.py."""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("GROQ_API_KEY"):
                    key = line.split("=", 1)[-1].strip().strip('"').strip("'")
                    if key:
                        return key
    return None

env_api_key = load_env_key()

# Session state 
if "df" not in st.session_state:
    st.session_state.df = None
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []
if "current_q" not in st.session_state:
    st.session_state.current_q = ""


# SIDEBAR

with st.sidebar:
    st.title("📊 AI Data Analyzer")
    st.caption("Accurate answers powered by Pandas code")

    st.divider()

    
    if env_api_key:
        groq_api_key = env_api_key 
    else:
        st.subheader("🔑 Groq API Key")
        groq_api_key = st.text_input(
            "Enter your key",
            type="password",
            placeholder="gsk_...",
            help="Or add GROQ_API_KEY=your_key to a .env file in the same folder"
        )
        if not groq_api_key:
            st.code("GROQ_API_KEY=gsk_your_key_here", language="text")
        st.divider()

    # File upload
    st.subheader("📁 Upload CSV")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.session_state.qa_history = []
        st.session_state.current_q = ""
        st.success(f"Loaded: {df.shape[0]} rows × {df.shape[1]} cols")

    # Show column list
    if st.session_state.df is not None:
        df = st.session_state.df
        st.divider()
        st.caption("**Columns in your file:**")
        for col in df.columns:
            dtype = "🔢" if pat.is_numeric_dtype(df[col]) else "🔤"
            st.caption(f"{dtype} {col}")

# MAIN — welcome screen if no file

if st.session_state.df is None:
    st.title("📊 AI Data Analyzer")
    st.markdown("Upload a CSV file in the sidebar to get started.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**📋 Summary**\nAuto stats, missing values, insights")
    with col2:
        st.info("**📊 Charts**\nInteractive Plotly visualizations")
    with col3:
        st.info("**💬 Ask**\nAccurate answers via Pandas code")

    st.stop()

df = st.session_state.df

# TABS

tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Summary",
    "📊 Charts",
    "💬 Ask a Question",
    "🤖 AI Insights",
])

# TAB 1 — Summary
with tab1:
    st.subheader("Dataset Summary")

    analysis = get_full_summary(df)
    info = analysis["basic_info"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{info['rows']:,}")
    c2.metric("Columns", info["columns"])
    c3.metric("Missing Values", sum(m["missing_count"] for m in analysis["missing"]))
    c4.metric("Size", f"{info['memory_kb']} KB")

    st.divider()

    st.subheader("Column Overview")
    col_rows = []
    for col in df.columns:
        kind = "Number" if pat.is_numeric_dtype(df[col]) else "Text"
        col_rows.append({
            "Column": col,
            "Type": kind,
            "Missing": int(df[col].isnull().sum()),
            "Unique Values": int(df[col].nunique()),
        })
    st.dataframe(pd.DataFrame(col_rows), width="stretch")

    st.divider()

    if analysis["numeric_stats"]:
        st.subheader("Numeric Column Statistics")
        stats_rows = []
        for col_name, s in analysis["numeric_stats"].items():
            stats_rows.append({
                "Column": col_name,
                "Mean": s["mean"], "Median": s["median"],
                "Std Dev": s["std"], "Min": s["min"], "Max": s["max"],
                "Outliers": s["outlier_count"],
            })
        st.dataframe(pd.DataFrame(stats_rows), width="stretch")

    if analysis["text_stats"]:
        st.subheader("Text Column Top Values")
        for col_name, t in analysis["text_stats"].items():
            with st.expander(f"{col_name} — {t['unique_count']} unique values"):
                top_df = pd.DataFrame(list(t["top_5"].items()), columns=["Value", "Count"])
                st.dataframe(top_df, width="stretch")

    if analysis["missing"]:
        st.subheader("Missing Values")
        st.dataframe(pd.DataFrame(analysis["missing"]), width="stretch")
    else:
        st.success("✅ No missing values found!")

    if analysis["auto_insights"]:
        st.subheader("Auto Insights")
        for insight in analysis["auto_insights"]:
            st.info(f"💡 {insight}")

    st.subheader("Data Preview (first 10 rows)")
    st.dataframe(df.head(10), width="stretch")


# TAB 2 — Charts

with tab2:
    st.subheader("Interactive Charts")

    numeric_cols = [c for c in df.columns if pat.is_numeric_dtype(df[c])]
    text_cols    = [c for c in df.columns if not pat.is_numeric_dtype(df[c])]

    if not numeric_cols:
        st.warning("No numeric columns found for charts.")
    else:
        chart_type = st.selectbox(
            "Chart type",
            ["Histogram", "Bar Chart", "Scatter Plot", "Box Plot", "Line Chart"]
        )

        if chart_type == "Histogram":
            col = st.selectbox("Column", numeric_cols)
            bins = st.slider("Bins", 10, 100, 30)
            fig = px.histogram(df, x=col, nbins=bins, title=f"Distribution of {col}")
            st.plotly_chart(fig, width="stretch")

        elif chart_type == "Bar Chart":
            if text_cols:
                x_col = st.selectbox("Category (X)", text_cols)
                y_col = st.selectbox("Value (Y)", numeric_cols)
                agg   = st.selectbox("Aggregate", ["mean", "sum", "count", "max", "min"])
                grouped = df.groupby(x_col)[y_col].agg(agg).reset_index().sort_values(y_col, ascending=False)
                fig = px.bar(grouped, x=x_col, y=y_col, title=f"{agg.title()} of {y_col} by {x_col}")
                st.plotly_chart(fig, width="stretch")
            else:
                st.warning("Need a text column for the X axis.")

        elif chart_type == "Scatter Plot":
            if len(numeric_cols) >= 2:
                x_col = st.selectbox("X axis", numeric_cols)
                y_col = st.selectbox("Y axis", numeric_cols, index=1)
                color = st.selectbox("Color by", ["None"] + text_cols)
                fig = px.scatter(df, x=x_col, y=y_col,
                                 color=None if color == "None" else color,
                                 title=f"{x_col} vs {y_col}", opacity=0.7)
                st.plotly_chart(fig, width="stretch")
            else:
                st.warning("Need at least 2 numeric columns.")

        elif chart_type == "Box Plot":
            y_col = st.selectbox("Value column", numeric_cols)
            x_col = st.selectbox("Group by", ["None"] + text_cols)
            fig = px.box(df, y=y_col, x=None if x_col == "None" else x_col,
                         title=f"Box Plot of {y_col}")
            st.plotly_chart(fig, width="stretch")

        elif chart_type == "Line Chart":
            y_col = st.selectbox("Y axis", numeric_cols)
            x_col = st.selectbox("X axis", ["Index"] + numeric_cols + text_cols)
            if x_col == "Index":
                fig = px.line(df, y=y_col, title=f"{y_col} over Index")
            else:
                fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} over {x_col}")
            st.plotly_chart(fig, width="stretch")

        if len(numeric_cols) >= 2:
            st.divider()
            st.subheader("Correlation Heatmap")
            corr = df[numeric_cols].corr().round(3)
            fig = px.imshow(corr, color_continuous_scale="RdBu_r",
                            zmin=-1, zmax=1, text_auto=True,
                            title="Correlation between numeric columns")
            st.plotly_chart(fig, width="stretch")


# TAB 3 — Ask a Question

with tab3:
    st.subheader("💬 Ask a Question")
    st.caption("The AI writes Pandas code and runs it on your actual data — always accurate.")

    if not groq_api_key:
        st.warning("⚠️ Add your Groq API key in the sidebar or .env file.")
    else:
        question = st.text_input(
            "Your question:",
            value=st.session_state.current_q,
            placeholder='e.g. "top 5 run scorers in 2021" or "average runs by team"',
        )

        if st.button("🔍 Get Answer", type="primary", disabled=not question):
            with st.spinner("Writing code and running it on your data..."):
                result = ask_question(question=question, df=df, api_key=groq_api_key)

            if result["error"]:
                st.error(f"Error: {result['error']}")
                st.code(result["code"], language="python")
            else:
                # RAW RESULT — always 100% accurate, straight from Pandas
                st.markdown("### ✅ Result (from your data)")
                if isinstance(result["result"], pd.DataFrame):
                    st.dataframe(result["result"].head(15), width="stretch")
                elif isinstance(result["result"], pd.Series):
                    st.dataframe(result["result"].head(15).reset_index(), width="stretch")
                else:
                    st.metric(label="Answer", value=str(result["result"]))

                st.caption(f"💬 AI explanation: {result['answer']}")

                with st.expander("🔍 See the Pandas code that ran"):
                    st.code(result["code"], language="python")

                st.session_state.qa_history.append({
                    "question": question,
                    "answer":   result["answer"],
                    "code":     result["code"],
                })
                st.session_state.current_q = ""

        if st.session_state.qa_history:
            st.divider()
            st.subheader("📜 Q&A History")
            for i, qa in enumerate(reversed(st.session_state.qa_history)):
                with st.expander(f"Q{len(st.session_state.qa_history)-i}: {qa['question'][:80]}"):
                    st.write(f"**Answer:** {qa['answer']}")
                    st.code(qa["code"], language="python")


# TAB 4 — AI Insights

with tab4:
    st.subheader("🤖 AI Insights")
    st.caption("The AI reads a summary of your data and highlights what's interesting.")

    if not groq_api_key:
        st.warning("⚠️ Add your Groq API key in the sidebar or .env file.")
    else:
        if st.button("✨ Generate Insights", type="primary"):
            with st.spinner("Analyzing your data..."):
                summary  = get_llm_summary(df)
                insights = get_ai_insights(summary, groq_api_key)

            lines = [l.strip() for l in insights.split("\n") if l.strip()]
            for line in lines:
                clean = line.lstrip("•-* ")
                if clean:
                    st.info(f"💡 {clean}")

        with st.expander("🔍 What the AI receives"):
            st.code(get_llm_summary(df), language="text")
