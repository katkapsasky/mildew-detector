# Import streamlit
import streamlit as st
from app_pages.multipage import MultiPage

# Load page scripts
from app_pages.page_summary import page_summary_body
from app_pages.page_cells_visualizer import page_leaf_visualizer_body
from app_pages.page_malaria_detector import page_mildew_detector_body
from app_pages.page_project_hypothesis import page_project_hypothesis_body
from app_pages.page_ml_performance import page_ml_performance_metrics

# Create an instance of the app
app = MultiPage(app_name= "Malaria Detector")  

# Add app pages using .add_page()
app.add_page("Quick Project Summary", page_summary_body)
app.add_page("Cells Visualizer", page_leaf_visualizer_body)
app.add_page("Malaria Detection", page_mildew_detector_body)
app.add_page("Project Hypothesis", page_project_hypothesis_body)
app.add_page("ML Performance Metrics", page_ml_performance_metrics)

# Run the app
app.run() 
