import streamlit as st
from dataloading import home_page
from analysis import analysis_page
from w_p import wellbeing_procrastination_page
from text_analysis import text_analysis_page


def main():
    """Main application function"""
    # Create sidebar for navigation
    st.sidebar.markdown("<h3 class='sidebar-content'>Navigation</h3>", unsafe_allow_html=True)
    
    # Navigation options
    page = st.sidebar.radio(
        "Select Section",
        ["Home", "Demographics & Study Habits", "Procrastination & Wellbeing", "Text Analysis"]
    )
    
    # Display appropriate page based on selection
    if page == "Home":
        home_page()
    elif page == "Demographics & Study Habits":
        analysis_page()
    elif page == "Procrastination & Wellbeing":
        wellbeing_procrastination_page()
    elif page == "Text Analysis":
        text_analysis_page()
    
    # App info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.info(
        """
        **Study on Habits and Routine of Students**
        
        A tool for analyzing student survey data to understand performance factors.
        
        - View demographic insights
        - Analyze study habits
        - Understand procrastination patterns
        - Extract insights from text responses
        """
    )
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666;">
          | Study on Habits and Routine of Students | Data-Driven Education Insights |
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()