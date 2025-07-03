PROJECT STRUCTURE : 

StudyPulse/
│
├── data/                  # Student CSVs or responses Receivesed from Google Form Link

├── docs/                  # Documentation, pptx and Google Form link

├── main.py                # Streamlit app

├── dataloading.py         # Data loading utilities

├── analysis.py            # Numeric/statistical analysis (CGPA or Study hours etc) 

├── student_responses_analysis.py  # Question-by-question analysis

├── text_analysis.py       # NLP-based text analysis of students views

├── w_p.py                 # Wellbeing and procrastination analysis

├── requirements.txt       # Python dependencies



INSTALLATION AND RUNNING :

git clone https://github.com/shantanuuh/StudyPulse.git
cd StudyPulse
pip install -r requirements.txt

streamlit run main.py

StudyPulse – A data-driven analysis of student habits, procrastination, and wellbeing using machine learning, sentiment analysis, and data visualization. Gain insights into academic performance and explore actionable recommendations for student success.