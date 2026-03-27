# app.py - New entry point for Streamlit UI

import streamlit as st
import os
import tempfile 
import shutil
# ❗ ADJUST IMPORT: Correct path based on your package structure
# Assuming 'src' is in the root and 'agent_eda' is the package name
from src.agent_eda.crew import DataAnalysisCrew 

st.set_page_config(layout="wide", page_title="Agent-Powered EDA Crew")
st.title("☕ EDA Agent")

uploaded_file = st.file_uploader("Upload your CSV dataset for analysis", type="csv")

if uploaded_file is not None:
    # 1. Create a secure, temporary directory for the uploaded file and crew output
    # This ensures the container's file system is used correctly.
    temp_dir = tempfile.mkdtemp()
    
    # Define the path where the crew will find the data and write outputs
    temp_csv_path = os.path.join(temp_dir, "uploaded_data.csv")
    
    # Write the uploaded file stream to the temporary file path
    with open(temp_csv_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.info(f"File '{uploaded_file.name}' ready for analysis.")
    st.write(f"Data Path for Crew: `{temp_csv_path}`")

    if st.button("Run Data Analysis Crew", type="primary"):
        with st.spinner("Agents are analyzing the data and preparing the report... this may take a few minutes."):
            try:
                # 2. Instantiate and Run the crew
                crew = DataAnalysisCrew()
                
                # Pass the path to the temporary file created from the upload
                final_report_markdown = crew.run(csv_path=temp_csv_path)

                # 3. Display the result
                st.subheader("✅ Final Analysis Report")
                st.markdown(final_report_markdown)
                
            except Exception as e:
                st.error(f"An error occurred during agent execution: {e}")
            finally:
                # 4. Cleanup the temporary directory
                st.write("Cleaning up temporary files...")
                shutil.rmtree(temp_dir)
                st.success("Cleanup complete. Analysis is ready.")