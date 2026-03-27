import os
from crewai import Agent, Crew, Process, Task, LLM
from crewai_tools import FileReadTool
#from langchain.tools import PythonREPLTool
#from langchain_experimental.tools import PythonREPLTool
#from crewai_tools import PythonREPLTool
#from langchain_experimental.tools.python.tool import PythonREPLTool
#from langchain.tools import PythonREPLTool
#from langchain_experimental.tools.python.tool import PythonREPLTool
#from langchain.tools import Tool
#from crewai_tools import FileReadTool, PythonREPLTool
from crewai_tools import CodeInterpreterTool
#from crewai.llm import AzureOpenAI
from dotenv import load_dotenv
load_dotenv()
# The FileReadTool is a convenient way for agents to read files.
# The PythonREPLTool allows agents to execute Python code.
# You must have langchain installed for the PythonREPLTool to work.
# pip install langchain

from crewai_tools import BaseTool

class FileWriteTool(BaseTool):
    name: str = "File Write Tool"
    description: str = "Writes content to a markdown file at a specific path."

    def _run(self, file_path: str, content: str) -> str:
        try:
            # Ensure the Outputs directory exists if you want to use it
            import os
            output_dir = os.path.dirname(file_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Successfully wrote report to {file_path}"
        except Exception as e:
            return f"Error writing file: {e}"

class DataAnalysisCrew:
    def __init__(self):
        # Ensure your API key is available as an environment variable
        # For Azure OpenAI, you need to set the following:
        
        DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
        AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
        AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
        # Here we use LLM from crewai, which will automatically pick up your Azure
        print(f"DEBUG: DEPLOYMENT_NAME is: {DEPLOYMENT_NAME}")
        print(f"DEBUG: AZURE_ENDPOINT is: {AZURE_ENDPOINT}")
        print(f"DEBUG: AZURE_API_KEY is: {'***' if AZURE_API_KEY else 'None'}")
        print(f"DEBUG: AZURE_API_VERSION is: {AZURE_API_VERSION}")

        # The error "argument of type 'NoneType' is not iterable" most often
        # means one of these is None.
        if not all([DEPLOYMENT_NAME, AZURE_ENDPOINT, AZURE_API_KEY, AZURE_API_VERSION]):
            raise ValueError("One or more Azure OpenAI environment variables are missing (None). Check your .env file and spelling.")
        # configuration from the environment variables.
        #self.llm = LLM(model="gpt-5-chat")
        #self.llm = LLM(model="azure/" + os.getenv("MODEL"))
        #self.llm = LLM(provider="azure_openai", model=os.getenv("MODEL"))
        #self.llm = AzureOpenAI(model="gpt-5-chat")
        self.llm = LLM(model=f"azure/{DEPLOYMENT_NAME}",api_base=AZURE_ENDPOINT,api_key=AZURE_API_KEY,api_version=AZURE_API_VERSION)
        
        self.file_read_tool = FileReadTool()
        self.code_interpreter_tool = CodeInterpreterTool()
        #self.python_repl_tool = PythonREPLTool()
        #self.python_repl_tool = Tool(name="Python REPL",description="A Python shell. Use this to execute Python commands. Input must be a valid Python command.",func=PythonREPLTool())

    def run(self, csv_path: str = 'data/sample.csv'):
        # Define agents
        data_analyst = Agent(
            role='Senior Data Analyst',
            goal='Find key insights and patterns from a given dataset.',
            backstory='You are a seasoned data analyst with a deep understanding of statistical methods and data manipulation. You excel at uncovering hidden truths in data and providing clear, actionable insights.',
            tools=[self.file_read_tool, self.code_interpreter_tool],
            verbose=True,
            allow_delegation=True,
            llm=self.llm
        )

        visualizer = Agent(
            role='Professional Data Visualizer',
            goal='Create clear and insightful visualizations from data analysis reports.',
            backstory='You are an expert at transforming complex data into easy-to-understand charts and graphs. Your work makes data stories compelling and accessible.',
            tools=[self.file_read_tool, self.code_interpreter_tool],
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )

        # Define tasks
        analysis_task = Task(
            description=f"Analyze the CSV file located at `{csv_path}`. Identify and document at least three key insights and trends.",
            expected_output="A comprehensive, markdown-formatted report detailing the key insights found in the data. This report MUST be saved to a file named 'Key_Insights_Report.md' within the 'outputs' directory.",
            agent=data_analyst
        )

        visualization_task = Task(
            description=f"Based on the analysis report from the Data Analyst, create a relevant visualization and save the chart as a PNG in the `outputs/` directory.",
            expected_output="A confirmation message that the chart has been saved to the `outputs/` directory.",
            agent=visualizer
        )

        # Instantiate the crew
        crew = Crew(
            agents=[data_analyst, visualizer],
            tasks=[analysis_task, visualization_task],
            process=Process.sequential,
            verbose=True
        )

        print("Crew is starting...")
        result = crew.kickoff()
        print("\n\n########################")
        print("## Crew Analysis Report")
        print("########################")
        print(result)