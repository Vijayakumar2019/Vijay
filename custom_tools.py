# custom_tools.py

from crewai.tools import BaseTool
import os

class FileWriteTool(BaseTool):
    # This name MUST be used EXACTLY in the YAML file
    name: str = "file_write_tool" 
    description: str = "Writes content to a text file at a specified path, creating the directory if it does not exist."
    
    # Define the inputs the agent must provide
    def _run(self, file_path: str, content: str) -> str:
        """Writes the given content to the file at file_path."""
        try:
            # 1. Ensure the directory exists
            output_dir = os.path.dirname(file_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # 2. Write the content to the file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return f"SUCCESS: Report saved to {file_path}"
        except Exception as e:
            return f"ERROR: Failed to write file: {e}"