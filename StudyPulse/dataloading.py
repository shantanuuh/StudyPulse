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
import io

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
    .sample-data-section {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin: 10px 0;
    }
    .download-button {
        margin: 10px 5px;
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

def generate_sample_data(size=100):
    """Generate a comprehensive sample dataset for student habits and routines"""
    np.random.seed(42)  # For reproducible results
    
    # Basic demographic data
    data = {
        "Student_ID": [f"STU{i:03d}" for i in range(1, size + 1)],
        "Gender": np.random.choice(["Male", "Female", "Non-binary", "Prefer not to say"], size, p=[0.45, 0.45, 0.05, 0.05]),
        "Age": np.random.randint(18, 30, size),
        "Current educational level": np.random.choice(["Bachelor's", "Master's", "PhD", "Other"], size, p=[0.6, 0.25, 0.1, 0.05]),
        "Field of study": np.random.choice(["Engineering", "Medicine", "Business", "Arts", "Science", "Social Sciences"], size),
        "Year of study": np.random.choice(["1st year", "2nd year", "3rd year", "4th year", "5+ years"], size),
        
        # Study habits
        "Hours of study per weekday": np.random.randint(1, 12, size),
        "Hours of study per weekend day": np.random.randint(0, 15, size),
        "Preferred study time": np.random.choice(["Early morning", "Morning", "Afternoon", "Evening", "Late night"], size),
        "Primary study location": np.random.choice(["Home", "Library", "Café", "Dorm room", "Study groups"], size),
        "Primary study resources": np.random.choice(["Textbooks", "Online courses", "Video lectures", "Study groups", "Notes"], size),
        
        # Procrastination and time management
        "Frequency of procrastination": np.random.choice(["Never", "Rarely", "Sometimes", "Often", "Always"], size, p=[0.05, 0.15, 0.3, 0.35, 0.15]),
        "What is the main reason you procrastinate?": np.random.choice([
            "Anxiety about performance", "Lack of interest in subject", "Social media distractions", 
            "Poor time management", "Overwhelming workload", "Perfectionism"
        ], size),
        
        # Health and wellbeing
        "Average Sleep hours": np.random.choice(["<5", "5-6", "7-8", ">8"], size, p=[0.15, 0.35, 0.4, 0.1]),
        "Exercise frequency per week": np.random.choice(["Never", "1-2 times", "3-4 times", "5+ times"], size, p=[0.2, 0.4, 0.25, 0.15]),
        "Stress level (1-10)": np.random.randint(1, 11, size),
        "Mental health rating (1-10)": np.random.randint(3, 11, size),
        
        # Distractions and environment
        "Main distraction while studying": np.random.choice([
            "Social media", "Friends/Family", "Games", "Internet browsing", "Noise", "Mobile notifications"
        ], size),
        "How often do household chores distract you from academics?": np.random.choice([
            "Never", "Rarely", "Sometimes", "Often", "Always"
        ], size, p=[0.1, 0.2, 0.4, 0.25, 0.05]),
        
        # Academic performance
        "Current GPA/Grade": np.random.choice(["A (90-100%)", "B (80-89%)", "C (70-79%)", "D (60-69%)", "F (<60%)"], size, p=[0.25, 0.35, 0.25, 0.1, 0.05]),
        "Satisfaction with academic performance": np.random.choice(["Very satisfied", "Satisfied", "Neutral", "Dissatisfied", "Very dissatisfied"], size),
    }
    
    # Text response columns with more realistic and varied responses
    study_routines = [
        "I wake up at 6 AM and study for 2 hours before breakfast. After classes, I review notes for 1 hour and do assignments in the evening.",
        "I prefer studying in the library from 9 AM to 5 PM with regular breaks every hour. I find the quiet environment helps me focus better.",
        "I'm a night owl, so I study from 10 PM to 2 AM when it's quiet. I take short naps during the day to compensate.",
        "I follow the Pomodoro technique - 25 minutes of focused study followed by 5-minute breaks. I repeat this cycle 4 times then take a longer break.",
        "I study in groups with classmates twice a week, then spend time alone reviewing and practicing problems.",
        "I don't have a fixed routine. I study when I feel motivated, which is usually irregular but intense sessions.",
        "I dedicate weekends to heavy studying and use weekdays for lighter review and assignment work.",
        "I study for 3-4 hours daily, taking breaks every 30 minutes. I alternate between different subjects to keep things interesting."
    ]
    
    procrastination_strategies = [
        "I use the Pomodoro technique to break large tasks into smaller, manageable chunks.",
        "I set specific deadlines for myself and reward myself with small treats when I meet them.",
        "I eliminate distractions by putting my phone in another room and using website blockers on my computer.",
        "I study in the library where seeing others work motivates me to stay focused.",
        "I ask friends to keep me accountable by checking in on my progress regularly.",
        "I start with the easiest tasks to build momentum before tackling harder ones.",
        "I use apps like Forest or Focus Keeper to track my study time and stay motivated.",
        "I remind myself of my long-term goals and why I'm studying to maintain motivation."
    ]
    
    academic_difficulties = [
        "I break down complex topics into smaller parts and use visual aids like mind maps and diagrams.",
        "I seek help from professors during office hours and form study groups with classmates.",
        "I use online resources like Khan Academy and YouTube tutorials to supplement my learning.",
        "I practice active recall by testing myself instead of just re-reading notes.",
        "I teach concepts to others as it helps me understand them better.",
        "I take regular breaks and don't hesitate to ask for help when I'm stuck.",
        "I use different learning methods like flashcards, summarizing, and practice problems.",
        "I prioritize getting enough sleep and exercise as they improve my cognitive function."
    ]
    
    stress_management = [
        "I practice meditation and deep breathing exercises to manage stress.",
        "I maintain a regular exercise routine which helps me release tension.",
        "I talk to friends and family about my concerns and seek their support.",
        "I ensure I get adequate sleep and maintain a healthy diet.",
        "I take regular breaks from studying to prevent burnout.",
        "I listen to calming music or nature sounds while studying.",
        "I practice time blocking to avoid last-minute cramming.",
        "I seek counseling services when stress becomes overwhelming."
    ]
    
    ideal_environment = [
        "A quiet library with natural lighting and comfortable seating.",
        "My room with good lighting, minimal distractions, and all my materials organized.",
        "A café with light background noise and a cup of coffee.",
        "A study group room where I can discuss concepts with peers.",
        "Outdoors in a peaceful setting like a park or garden.",
        "A dedicated study space at home with ergonomic furniture.",
        "The university study hall with access to resources and help.",
        "A well-ventilated room with plants and inspiring decorations."
    ]
    
    break_activities = [
        "I scroll through social media, watch short videos, or chat with friends.",
        "I take walks outside or do light stretching exercises.",
        "I listen to music or podcasts while having a snack.",
        "I play mobile games or browse the internet for entertainment.",
        "I meditate or practice mindfulness exercises.",
        "I do household chores or organize my study space.",
        "I call family or friends to catch up and socialize.",
        "I read fiction books or watch TV shows to relax."
    ]
    
    # Add text columns with varied responses
    data["Describe your typical study routine in detail."] = [np.random.choice(study_routines) for _ in range(size)]
    data["What strategies do you use to overcome procrastination?"] = [np.random.choice(procrastination_strategies) for _ in range(size)]
    data["What strategies help you overcome academic difficulties?"] = [np.random.choice(academic_difficulties) for _ in range(size)]
    data["How do you deal with academic stress?"] = [np.random.choice(stress_management) for _ in range(size)]
    data["Describe your ideal learning environment."] = [np.random.choice(ideal_environment) for _ in range(size)]
    data["How do you spend your break time?"] = [np.random.choice(break_activities) for _ in range(size)]
    
    # Add some satisfaction ratings
    data["How satisfied are you with your time management skills?"] = np.random.choice([
        "Very satisfied", "Satisfied", "Neutral", "Dissatisfied", "Very dissatisfied"
    ], size)
    
    return pd.DataFrame(data)

def convert_df_to_excel(df):
    """Convert DataFrame to Excel format for download"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Student_Survey_Data', index=False)
        
        # Get workbook and worksheet
        workbook = writer.book
        worksheet = writer.sheets['Student_Survey_Data']
        
        # Add header formatting
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'bg_color': '#4CAF50',
            'font_color': 'white'
        })
        
        # Apply header formatting
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
            
        # Auto-adjust column width
        for col_num, col_name in enumerate(df.columns):
            max_length = max(df[col_name].astype(str).apply(len).max(), len(col_name))
            worksheet.set_column(col_num, col_num, min(max_length + 2, 50))
    
    output.seek(0)
    return output.getvalue()

def convert_df_to_csv(df):
    """Convert DataFrame to CSV format for download"""
    return df.to_csv(index=False).encode('utf-8')

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
    
    # Main upload section
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
    
    # How to use section
    st.markdown("---")
    st.markdown("### 📋 How to use:")
    st.markdown("""
    1. **Upload** your student survey data (CSV or Excel format)
    2. **Review** the data preview to ensure it loaded correctly
    3. **Navigate** through analysis sections using the sidebar
    4. **Download** insights and visualizations from each section
    """)
    
    # Enhanced sample data section
    st.markdown("---")
    st.markdown("<div class='sample-data-section'>", unsafe_allow_html=True)
    st.markdown("### 📊 No data to try?")
    st.markdown("**Get started with our comprehensive sample dataset!**")
    
    # Generate sample data for preview
    sample_df = generate_sample_data(100)
    
    st.markdown(f"""
    Our sample dataset includes:
    - **{len(sample_df)} student responses**
    - **{len(sample_df.columns)} data points per student**
    - Demographics, study habits, procrastination patterns
    - Detailed text responses for NLP analysis
    - Health and wellbeing indicators
    """)
    
    # Download buttons row
    st.markdown("**📥 Download Sample Data:**")
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        # Excel download
        excel_data = convert_df_to_excel(sample_df)
        st.download_button(
            label="📥 Download Excel",
            data=excel_data,
            file_name="student_survey_sample_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Download sample data in Excel format",
            use_container_width=True
        )
    
    with col2:
        # CSV download
        csv_data = convert_df_to_csv(sample_df)
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name="student_survey_sample_data.csv",
            mime="text/csv",
            help="Download sample data in CSV format",
            use_container_width=True
        )
    
    with col3:
        # Try sample data button
        if st.button("🚀 Try Sample Data", help="Load sample data directly into the application", use_container_width=True):
            st.session_state.data = sample_df
            st.session_state.preprocessing_complete = True
            st.session_state.selected_columns = [
                "Describe your typical study routine in detail.",
                "What strategies do you use to overcome procrastination?",
                "What strategies help you overcome academic difficulties?",
                "How do you deal with academic stress?",
                "Describe your ideal learning environment.",
                "How do you spend your break time?"
            ]
            
            st.success("✅ Sample dataset loaded successfully!")
            st.balloons()
            
            # Show quick stats
            st.markdown("**Quick Stats:**")
            st.markdown(f"- Students: {len(sample_df)}")
            st.markdown(f"- Text responses: {len(st.session_state.selected_columns)}")
            st.markdown(f"- Ready for analysis! 🎯")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Show sample data preview
    with st.expander("👀 Preview Sample Data"):
        st.dataframe(sample_df.head(5), use_container_width=True)
        
        # Show column information in a more organized way
        st.markdown("---")
        st.markdown("**📊 Dataset Categories:**")
        
        categories = {
            "👤 Demographics": ["Student_ID", "Gender", "Age", "Current educational level", "Field of study", "Year of study"],
            "📚 Study Habits": ["Hours of study per weekday", "Hours of study per weekend day", "Preferred study time", "Primary study location", "Primary study resources"],
            "⏰ Procrastination": ["Frequency of procrastination", "What is the main reason you procrastinate?"],
            "🏥 Health & Wellbeing": ["Average Sleep hours", "Exercise frequency per week", "Stress level (1-10)", "Mental health rating (1-10)"],
            "🎯 Academic Performance": ["Current GPA/Grade", "Satisfaction with academic performance"],
            "💭 Text Responses": ["Describe your typical study routine in detail.", "What strategies do you use to overcome procrastination?", "What strategies help you overcome academic difficulties?", "How do you deal with academic stress?"]
        }
        
        # Display categories in a grid
        for i, (category, columns) in enumerate(categories.items()):
            available_cols = [col for col in columns if col in sample_df.columns]
            if available_cols:
                with st.container():
                    st.markdown(f"**{category}** ({len(available_cols)} columns)")
                    st.markdown(f"*{', '.join(available_cols[:4])}{'...' if len(available_cols) > 4 else ''}*")
                    st.markdown("")
