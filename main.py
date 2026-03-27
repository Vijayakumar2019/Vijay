#!/usr/bin/env python
import os
from agent_eda.crew import DataAnalysisCrew

def run():
    """
    Entry point to run the DataAnalysisCrew.
    You can override the CSV path using an environment variable or edit directly here.
    """
    # Set working directory to project root
    os.chdir(r"C:\Projects\agent_eda")
    # Default CSV file path
    csv_file_path = os.getenv("CSV_FILE_PATH", r"C:\Projects\agent_eda\data\sample.csv")

    crew = DataAnalysisCrew()
    crew.run(csv_path=csv_file_path)

if __name__ == "__main__":
    run()
