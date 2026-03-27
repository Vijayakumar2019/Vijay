import os
import yaml
from crewai import Agent, Crew, Process, Task, LLM
from crewai_tools import FileReadTool, CodeInterpreterTool
from dotenv import load_dotenv

load_dotenv()

class DataAnalysisCrew:
    def __init__(self):
        # Detect correct base and config paths
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_dir = os.path.join(base_dir, "config")

        # ✅ Load YAML configs
        self.agents_config = self._load_yaml(os.path.join(config_dir, "agents.yaml")).get("agents", [])
        self.tasks_config = self._load_yaml(os.path.join(config_dir, "tasks.yaml")).get("tasks", [])

        # ✅ Load Azure OpenAI Environment Variables
        DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
        AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
        AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

        # Debug prints (optional)
        print(f"\n🔹 DEPLOYMENT_NAME: {DEPLOYMENT_NAME}")
        print(f"🔹 AZURE_ENDPOINT: {AZURE_ENDPOINT}")
        print(f"🔹 AZURE_API_VERSION: {AZURE_API_VERSION}\n")

        if not all([DEPLOYMENT_NAME, AZURE_ENDPOINT, AZURE_API_KEY, AZURE_API_VERSION]):
            raise EnvironmentError(
                "❌ Missing one or more Azure OpenAI environment variables. Check your .env file."
            )

        # ✅ Initialize Azure OpenAI LLM connection
        self.llm = LLM(
            model=f"azure/{DEPLOYMENT_NAME}",
            api_base=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            api_version=AZURE_API_VERSION
        )

        # ✅ Tools
        self.file_read_tool = FileReadTool()
        self.code_interpreter_tool = CodeInterpreterTool()

    def _load_yaml(self, path: str):
        """Safely load a YAML file"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"⚠️ Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def run(self, csv_path: str = "data/sample.csv"):
        """Run the CrewAI pipeline"""
        agents = []
        for a in self.agents_config:
            agents.append(
                Agent(
                    role=a["role"],
                    goal=a["goal"],
                    backstory=a["backstory"],
                    tools=[self.file_read_tool, self.code_interpreter_tool],
                    verbose=True,
                    allow_delegation=a.get("allow_delegation", False),
                    llm=self.llm,
                )
            )

        tasks = []
        for t, agent in zip(self.tasks_config, agents):
            desc = t["description"].replace("{csv_path}", csv_path)
            tasks.append(
                Task(
                    description=desc,
                    expected_output=t["expected_output"],
                    agent=agent,
                )
            )

        crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
        )

        print("\n🚀 Crew is starting...")
        result = crew.kickoff()

        print("\n\n########################")
        print("## Crew Analysis Report")
        print("########################")
        print(result)
