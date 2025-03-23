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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Study on Habits and Routine of Students",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        color: #1E88E5;
        font-size: 2.5rem;
        text-align: center;
    }
    .sub-header {
        color: #26A69A;
        font-size: 1.8rem;
    }
    .stExpander {
        border-left: 2px solid #1E88E5;
        padding-left: 10px;
    }
    .stProgress .st-bo {
        background-color: #26A69A;
    }
    .sidebar-content {
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize NLTK resources with progress indication
@st.cache_resource
def download_nltk_resources():
    with st.spinner("Loading language resources..."):
        nltk.download('punkt')
        nltk.download('stopwords')
        return True

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'text_analysis_results' not in st.session_state:
    st.session_state.text_analysis_results = None
if 'selected_columns' not in st.session_state:
    st.session_state.selected_columns = []
if 'preprocessing_complete' not in st.session_state:
    st.session_state.preprocessing_complete = False
if 'progress' not in st.session_state:
    st.session_state.progress = 0

# Common text columns for analysis - can be dynamically adjusted based on data
DEFAULT_TEXT_COLUMNS = [
    "Describe your typical study routine in detail.",
    "How satisfied are you with your time management skills?",
    "What strategies help you overcome academic difficulties?",
    "How do you deal with academic stress?",
    "Describe your ideal learning environment.",
    "What is the main reason you procrastinate?",
    "What strategies do you use to overcome procrastination?",
    "How has procrastination impacted your academic performance, and what did you learn?",
    "How do you spend your break time?"
]

# Cache sentiment analyzer to avoid reloading
@st.cache_resource
def load_sentiment_analyzer():
    try:
        with st.spinner("Loading sentiment analysis model..."):
            return pipeline("sentiment-analysis")
    except Exception as e:
        logger.error(f"Error loading sentiment analyzer: {e}")
        st.error("Failed to load sentiment analysis model. Some features may be unavailable.")
        return None

# Data loading and preprocessing
def load_data(uploaded_file):
    """Load and preprocess the data from uploaded file"""
    try:
        # Display a progress bar for data loading
        progress_bar = st.progress(0)
        
        # Load the data based on file type
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload CSV or Excel files.")
            return None
        
        progress_bar.progress(50)
        
        # Basic preprocessing
        # Convert column names to string to avoid potential errors
        df.columns = df.columns.astype(str)
        
        # Remove duplicate rows
        initial_rows = len(df)
        df = df.drop_duplicates()
        if len(df) < initial_rows:
            st.info(f"Removed {initial_rows - len(df)} duplicate entries.")
        
        # Check for missing values
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            with st.expander("Missing Data Information"):
                st.write("Missing values in each column:")
                st.write(missing_data[missing_data > 0])
        
        progress_bar.progress(100)
        time.sleep(0.5)  # Show completed progress for a moment
        progress_bar.empty()
        
        # Identify potential text columns for analysis
        potential_text_cols = []
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if column contains text responses (more than just single words)
                sample = df[col].dropna().astype(str).str.split().str.len()
                if len(sample) > 0 and sample.mean() > 3:  # Average more than 3 words
                    potential_text_cols.append(col)
        
        st.session_state.selected_columns = [col for col in DEFAULT_TEXT_COLUMNS if col in potential_text_cols]
        st.session_state.preprocessing_complete = True
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        st.error(f"Error loading data: {str(e)}")
        return None

def home_page():
    """Render home page with data loading and options"""
    st.markdown("<h1 class='main-header'>Study on Habits and Routine of Students</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Upload your student survey data to gain insights on:
        - Student demographics and study habits
        - Procrastination patterns and strategies
        - Text response analysis using NLP
        - Wellbeing and distraction factors
        """)
        
        uploaded_file = st.file_uploader("Upload Student Data", type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file:
            df = load_data(uploaded_file)
            
            if df is not None:
                st.session_state.data = df
                st.success("Data loaded successfully!")
                
                # Display data preview in an expander
                with st.expander("Data Preview"):
                    st.dataframe(df.head(), use_container_width=True)
                    
                    # Show data statistics
                    st.subheader("Dataset Statistics")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Rows:** {df.shape[0]}")
                        st.write(f"**Columns:** {df.shape[1]}")
                    with col2:
                        numeric_cols = df.select_dtypes(include=['number']).columns
                        categorical_cols = df.select_dtypes(include=['object']).columns
                        st.write(f"**Numeric columns:** {len(numeric_cols)}")
                        st.write(f"**Categorical/Text columns:** {len(categorical_cols)}")
                
                # Show available analyses
                st.subheader("Available Analyses")
                st.info("Use the sidebar to navigate to different analysis sections.")
    
    with col2:
        st.markdown("""
        ### How to use:
        1. Upload your student survey data (CSV or Excel)
        2. Review the data preview 
        3. Navigate through analysis sections using the sidebar
        4. Download insights and visualizations
        """)
        
        # Add example dataset option
        st.markdown("---")
        st.markdown("### No data to try?")
        if st.button("Load Example Dataset"):
            # Create a simple example dataset
            example_data = {
                "Gender": np.random.choice(["Male", "Female", "Non-binary"], 100),
                "Age": np.random.randint(18, 30, 100),
                "Current educational level": np.random.choice(["Bachelor's", "Master's", "PhD"], 100),
                "Hours of study per weekday": np.random.randint(1, 10, 100),
                "Primary study resources": np.random.choice(["Textbooks", "Online courses", "Video lectures", "Study groups"], 100),
                "What is the main reason you procrastinate?": np.random.choice(["Anxiety", "Lack of interest", "Distractions", "Poor time management"], 100),
                "Average Sleep hours": np.random.choice(["<5", "5-6", "7-8", ">8"], 100),
                "Exercise frequency per week": np.random.choice(["Never", "1-2 times", "3-4 times", "5+ times"], 100),
                "Distraction while studying": np.random.choice(["Social media", "Friends", "Games", "Family"], 100),
                "How often do household chores distract you from academics?": np.random.choice(["Never", "Rarely", "Sometimes", "Often", "Always"], 100),
            }
            
            # Add text columns with more realistic responses
            study_routines = [
                "I study for 2 hours in the morning and 3 hours at night. I take short breaks every 30 minutes.",
                "I prefer studying in the library from 9am to 5pm with lunch breaks.",
                "I study only when I feel motivated, usually late at night.",
                "I follow a strict pomodoro technique with 25 minute sessions.",
                "I prefer group study sessions followed by individual practice."
            ]
            
            procrastination_strategies = [
                "I use the pomodoro technique to break down tasks.",
                "I set clear deadlines and rewards for myself.",
                "I try to eliminate distractions by turning off notifications.",
                "I work in the library where others are studying.",
                "I ask friends to keep me accountable."
            ]
            
            example_data["Describe your typical study routine in detail."] = [
                np.random.choice(study_routines) for _ in range(100)
            ]
            
            example_data["What strategies do you use to overcome procrastination?"] = [
                np.random.choice(procrastination_strategies) for _ in range(100)
            ]
            
            df = pd.DataFrame(example_data)
            st.session_state.data = df
            st.session_state.preprocessing_complete = True
            st.session_state.selected_columns = [
                "Describe your typical study routine in detail.",
                "What strategies do you use to overcome procrastination?"
            ]
            
            st.success("Example dataset loaded! Navigate to analysis sections to explore.")
            
            # Display data preview
            with st.expander("Example Data Preview"):
                st.dataframe(df.head())

