import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def analysis_page():
    """Render data analysis page with demographic insights"""
    st.markdown("<h2 class='sub-header'>📊 Demographic & Study Habits Analysis</h2>", unsafe_allow_html=True)

    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first on the Home page!")
        return

    df = st.session_state.data

    # ✅ Column selector for analysis
    demo_columns = st.multiselect(
        "📌 Select columns to analyze:", 
        options=df.columns.tolist(),
        default=[col for col in ['Gender', 'Age', 'Current educational level'] if col in df.columns]
    )

    if not demo_columns:
        st.info("ℹ️ Please select at least one column to analyze.")
        return

    # ✅ Demographic Analysis
    st.subheader("📊 Demographic Analysis")

    cols = st.columns(min(3, len(demo_columns)))

    for i, column in enumerate(demo_columns):
        col_index = i % len(cols)

        with cols[col_index]:
            try:
                if df[column].dtype == 'object' or df[column].nunique() < 10:
                    # Categorical data - pie or bar chart
                    value_counts = df[column].value_counts()
                    fig = px.pie(
                        values=value_counts.values, 
                        names=value_counts.index, 
                        title=f"{column} Distribution",
                        color_discrete_sequence=px.colors.qualitative.Set3
                    ) if len(value_counts) <= 5 else px.bar(
                        x=value_counts.index, 
                        y=value_counts.values,
                        title=f"{column} Distribution",
                        labels={'x': column, 'y': 'Count'},
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                else:
                    # Numerical data - histogram
                    fig = px.histogram(
                        df, 
                        x=column, 
                        title=f"{column} Distribution",
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )

                fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)

                # ✅ Key statistics
                st.write(f"**{column} Statistics:**")
                if df[column].dtype in ['int64', 'float64']:
                    stats = df[column].describe()
                    st.write(f"Mean: {stats['mean']:.2f}, Median: {stats['50%']:.2f}")
                else:
                    st.write(f"Most common: {df[column].value_counts().idxmax()}")

            except Exception as e:
                st.error(f"❌ Error creating chart for {column}: {str(e)}")

    # ✅ Study Habits Section
    st.subheader("📚 Study Habits Analysis")

    habit_columns = st.multiselect(
        "📌 Select study habit columns to analyze:",
        options=df.columns.tolist(),
        default=[col for col in ['Hours of study per weekday', 'Primary study resources'] if col in df.columns]
    )

    if habit_columns:
        cols = st.columns(min(2, len(habit_columns)))

        for i, column in enumerate(habit_columns):
            with cols[i % len(cols)]:
                try:
                    fig = px.box(df, y=column, title=f"{column}", color_discrete_sequence=px.colors.qualitative.Pastel1) \
                        if df[column].dtype in ['int64', 'float64'] else \
                        px.bar(x=df[column].value_counts().index, y=df[column].value_counts().values,
                               title=f"{column}", labels={'x': column, 'y': 'Count'},
                               color_discrete_sequence=px.colors.qualitative.Pastel1)

                    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"❌ Error creating chart for {column}: {str(e)}")

    # ✅ Correlation Analysis (Now Supports Text Columns)
    with st.expander("📊 Correlation Analysis"):
        # ✅ Convert categorical columns to numerical using Label Encoding
        encoded_df = df.copy()
        label_encoders = {}

        for col in df.select_dtypes(include=['object']).columns:
            label_encoders[col] = LabelEncoder()
            encoded_df[col] = label_encoders[col].fit_transform(df[col].astype(str))

        numeric_cols = encoded_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numeric_cols = [col for col in numeric_cols if encoded_df[col].nunique() > 1]  # Remove constant columns

        selected_cols = []  # ✅ Initialize to prevent "UnboundLocalError"

        if len(numeric_cols) >= 2:
            selected_cols = st.multiselect(
                "🔹 Select columns for correlation analysis:", 
                options=numeric_cols,
                default=numeric_cols[:min(5, len(numeric_cols))]
            )

            if len(selected_cols) >= 2:
                df_cleaned = encoded_df[selected_cols].fillna(0)
                correlation = df_cleaned.corr()

                fig = px.imshow(
                    correlation.values,
                    labels=dict(color="Correlation"),
                    x=correlation.columns,
                    y=correlation.columns,
                    text_auto=".2f",
                    color_continuous_scale="RdBu_r",
                    title="🔍 Correlation Matrix"
                )
                st.plotly_chart(fig, use_container_width=True)

                strong_correlations = [
                    f"**{correlation.columns[i]}** & **{correlation.columns[j]}**: {correlation.iloc[i, j]:.2f}"
                    for i in range(len(correlation.columns))
                    for j in range(i+1, len(correlation.columns))
                    if abs(correlation.iloc[i, j]) > 0.5
                ]

                st.subheader("🔗 Strong Correlations Detected:") if strong_correlations else st.write("✅ No strong correlations found.")
                for corr in strong_correlations:
                    st.write(f"- {corr}")
            else:
                st.info("⚠️ Please select at least two columns for correlation analysis.")
        else:
            st.info("⚠️ Not enough numeric columns available for correlation analysis.")

