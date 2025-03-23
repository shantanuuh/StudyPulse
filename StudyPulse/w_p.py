import streamlit as st
import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from transformers import pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from collections import Counter
import re
import time
import logging
from functools import lru_cache


def wellbeing_procrastination_page():
    """Analyze procrastination, wellbeing, and distractions"""
    st.markdown("<h2 class='sub-header'>Procrastination & Wellbeing Analysis</h2>", unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("Please upload data first on the Home page!")
        return
    
    df = st.session_state.data
    
    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["Procrastination", "Wellbeing", "Distractions"])
    
    with tab1:
        st.subheader("Procrastination Analysis")
        
        # Find procrastination-related columns
        procrastination_cols = [col for col in df.columns if 'procrastinat' in col.lower()]
        
        if procrastination_cols:
            cols = st.columns(min(2, len(procrastination_cols)))
            
            for i, column in enumerate(procrastination_cols):
                with cols[i % len(cols)]:
                    if df[column].dtype == 'object':
                        try:
                            # Text processing for better visualization
                            if df[column].str.len().mean() > 100:  # Long text responses
                                st.write(f"**{column}**")
                                # Extract keywords
                                words = " ".join(df[column].dropna().astype(str))
                                words = re.sub(r'\W+', ' ', words.lower())
                                stop_words = set(stopwords.words('english'))
                                word_tokens = word_tokenize(words)
                                filtered_words = [w for w in word_tokens if w not in stop_words and len(w) > 2]
                                word_freq = Counter(filtered_words)
                                
                                # Show most common words
                                top_words = word_freq.most_common(10)
                                
                                # Create bar chart of top words
                                fig = px.bar(
                                    x=[word[0] for word in top_words],
                                    y=[word[1] for word in top_words],
                                    title=f"Most Common Words in '{column}'",
                                    labels={'x': 'Words', 'y': 'Frequency'},
                                    color_discrete_sequence=px.colors.qualitative.Pastel1
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Sample responses
                                with st.expander("Sample Responses"):
                                    for response in df[column].dropna().sample(min(5, len(df[column].dropna()))).values:
                                        st.write(f"- {response}")
                            else:
                                # Shorter responses - categorize and count
                                value_counts = df[column].value_counts()
                                
                                if len(value_counts) <= 7:  # Use pie chart for fewer categories
                                    fig = px.pie(
                                        values=value_counts.values,
                                        names=value_counts.index,
                                        title=f"{column}",
                                        color_discrete_sequence=px.colors.qualitative.Pastel1
                                    )
                                else:  # Use bar chart for more categories
                                    fig = px.bar(
                                        x=value_counts.index[:10],  # Limit to top 10
                                        y=value_counts.values[:10],
                                        title=f"Top Responses: {column}",
                                        labels={'x': column, 'y': 'Count'},
                                        color_discrete_sequence=px.colors.qualitative.Pastel1
                                    )
                                st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error analyzing column {column}: {str(e)}")
                    else:
                        # Numeric data
                        fig = px.histogram(
                            df, 
                            x=column,
                            title=f"{column} Distribution",
                            color_discrete_sequence=px.colors.qualitative.Pastel1
                        )
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No columns with 'procrastination' found in the data. Try uploading a different dataset or manually select columns for analysis.")
            
            # Manual column selection option
            manual_cols = st.multiselect(
                "Select columns related to procrastination:",
                options=df.columns
            )
            
            if manual_cols:
                for column in manual_cols:
                    st.write(f"**{column}**")
                    if df[column].dtype == 'object':
                        value_counts = df[column].value_counts()
                        fig = px.bar(
                            x=value_counts.index[:10],
                            y=value_counts.values[:10],
                            title=f"{column}",
                            color_discrete_sequence=px.colors.qualitative.Pastel1
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        fig = px.histogram(
                            df,
                            x=column,
                            title=f"{column}",
                            color_discrete_sequence=px.colors.qualitative.Pastel1
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Wellbeing Analysis")
        
        # Find wellbeing-related columns
        wellbeing_cols = [col for col in df.columns if any(term in col.lower() for term in 
                                                         ['sleep', 'stress', 'exercise', 'health', 
                                                          'anxiety', 'wellbeing', 'well-being'])]
        
        if wellbeing_cols:
            cols = st.columns(min(2, len(wellbeing_cols)))
            
            for i, column in enumerate(wellbeing_cols):
                with cols[i % len(cols)]:
                    try:
                        if df[column].dtype in ['int64', 'float64']:
                            fig = px.histogram(
                                df,
                                x=column,
                                title=f"{column} Distribution",
                                color_discrete_sequence=px.colors.qualitative.Pastel2
                            )
                        else:
                            value_counts = df[column].value_counts()
                            fig = px.pie(
                                values=value_counts.values,
                                names=value_counts.index,
                                title=f"{column}",
                                color_discrete_sequence=px.colors.qualitative.Pastel2
                            )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error analyzing column {column}: {str(e)}")
        else:
            st.info("No wellbeing-related columns found. Try uploading a different dataset or manually select columns for analysis.")
            
            # Manual column selection
            manual_cols = st.multiselect(
                "Select columns related to wellbeing:",
                options=df.columns
            )
            
            if manual_cols:
                for column in manual_cols:
                    try:
                        if df[column].dtype in ['int64', 'float64']:
                            fig = px.histogram(
                                df,
                                x=column,
                                title=f"{column}",
                                color_discrete_sequence=px.colors.qualitative.Pastel2
                            )
                        else:
                            value_counts = df[column].value_counts()
                            fig = px.pie(
                                values=value_counts.values,
                                names=value_counts.index,
                                title=f"{column}",
                                color_discrete_sequence=px.colors.qualitative.Pastel2
                            )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error analyzing column {column}: {str(e)}")
    
    with tab3:
        st.subheader("Distractions Analysis")
        
        # Find distraction-related columns
        distraction_cols = [col for col in df.columns if any(term in col.lower() for term in 
                                                          ['distract', 'focus', 'attention', 
                                                           'concentration', 'interrupt'])]
        
        if distraction_cols:
            cols = st.columns(min(2, len(distraction_cols)))
            
            for i, column in enumerate(distraction_cols):
                with cols[i % len(cols)]:
                    try:
                        if df[column].dtype in ['int64', 'float64']:
                            fig = px.histogram(
                                df,
                                x=column,
                                title=f"{column}",
                                color_discrete_sequence=px.colors.qualitative.Pastel
                            )
                        else:
                            value_counts = df[column].value_counts()
                            fig = px.bar(
                                x=value_counts.index,
                                y=value_counts.values,
                                title=f"{column}",
                                color_discrete_sequence=px.colors.qualitative.Pastel
                            )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error analyzing column {column}: {str(e)}")
        else:
            st.info("No distraction-related columns found. Try uploading a different dataset or manually select columns for analysis.")
            
            # Manual column selection
            manual_cols = st.multiselect(
                "Select columns related to distractions:",
                options=df.columns
            )
            
            if manual_cols:
                for column in manual_cols:
                    try:
                        if df[column].dtype in ['int64', 'float64']:
                            fig = px.histogram(
                                df,
                                x=column,
                                title=f"{column}",
                                color_discrete_sequence=px.colors.qualitative.Pastel
                            )
                        else:
                            value_counts = df[column].value_counts()
                            fig = px.bar(
                                x=value_counts.index,
                                y=value_counts.values,
                                title=f"{column}",
                                color_discrete_sequence=px.colors.qualitative.Pastel
                            )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error analyzing column {column}: {str(e)}")
    
    # Advanced analysis - Relationships
    with st.expander("Advanced Analysis: Relationships Between Factors"):
        st.write("Explore relationships between procrastination, wellbeing, and academic performance.")
        
        # Column selection for relationship analysis
        col1, col2 = st.columns(2)
        
        with col1:
            x_axis = st.selectbox("Select X-axis variable:", df.columns)
        
        with col2:
            y_axis = st.selectbox("Select Y-axis variable:", 
                                 [col for col in df.columns if col != x_axis])
        
        if x_axis and y_axis:
            try:
                # Create appropriate plot based on data types
                if df[x_axis].dtype in ['int64', 'float64'] and df[y_axis].dtype in ['int64', 'float64']:
                    # Both numerical - scatter plot
                    fig = px.scatter(
                        df,
                        x=x_axis,
                        y=y_axis,
                        title=f"Relationship between {x_axis} and {y_axis}",
                        trendline="ols",  # Add trend line
                        color_discrete_sequence=px.colors.qualitative.Set1
                    )
                    
                    # Calculate correlation
                    correlation = df[[x_axis, y_axis]].corr().iloc[0,1]
                    st.write(f"**Correlation coefficient:** {correlation:.3f}")
                    
                elif df[x_axis].dtype in ['int64', 'float64'] and df[y_axis].dtype not in ['int64', 'float64']:
                    # X numeric, Y categorical - box plot
                    fig = px.box(
                        df,
                        x=y_axis,
                        y=x_axis,
                        title=f"{x_axis} by {y_axis}",
                        color_discrete_sequence=px.colors.qualitative.Set1
                    )
                    
                elif df[x_axis].dtype not in ['int64', 'float64'] and df[y_axis].dtype in ['int64', 'float64']:
                    # X categorical, Y numeric - box plot
                    fig = px.box(
                        df,
                        x=x_axis,
                        y=y_axis,
                        title=f"{y_axis} by {x_axis}",
                        color_discrete_sequence=px.colors.qualitative.Set1
                    )
                    
                else:
                    # Both categorical - heatmap
                    crosstab = pd.crosstab(df[x_axis], df[y_axis])
                    fig = px.imshow(
                        crosstab,
                        title=f"Relationship between {x_axis} and {y_axis}",
                        color_continuous_scale='Blues'
                    )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add statistical insight based on plot type
                if df[x_axis].dtype in ['int64', 'float64'] and df[y_axis].dtype in ['int64', 'float64']:
                    # Interpret correlation
                    if abs(correlation) < 0.3:
                        st.write("There appears to be a weak relationship between these variables.")
                    elif abs(correlation) < 0.7:
                        st.write("There appears to be a moderate relationship between these variables.")
                    else:
                        st.write("There appears to be a strong relationship between these variables.")
                else:
                    st.write("Examine the chart to identify patterns in how these categories relate.")
                
            except Exception as e:
                st.error(f"Error creating relationship chart: {str(e)}")