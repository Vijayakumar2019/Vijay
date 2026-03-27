# data_analysis_crew.py

import os
import yaml
from crewai import Agent, Crew, Process, Task, LLM
from crewai_tools import FileReadTool, CodeInterpreterTool
from dotenv import load_dotenv

# 👇 Import the custom tool
from .custom_tools import FileWriteTool 

load_dotenv()

class DataAnalysisCrew:
    def __init__(self):
        # Detect correct base and config paths
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_dir = os.path.join(base_dir, "config") # Assuming config is a sub-directory

        # Load YAML configs
        self.agents_config = self._load_yaml(os.path.join(config_dir, "agents.yaml")).get("agents", [])
        self.tasks_config = self._load_yaml(os.path.join(config_dir, "tasks.yaml")).get("tasks", [])

        # Load Azure OpenAI Environment Variables
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

        # Initialize Azure OpenAI LLM connection
        self.llm = LLM(
            model=f"azure/{DEPLOYMENT_NAME}",
            api_base=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            api_version=AZURE_API_VERSION
        )

        # 👇 🛠️ Initialize ALL available tool instances and map them to their YAML names
        self.available_tools = {
            "file_read_tool": FileReadTool(),
            "code_interpreter_tool": CodeInterpreterTool(),
            "file_write_tool": FileWriteTool(), # <-- Custom tool added here
        }

    def _load_yaml(self, path: str):
        """Safely load a YAML file"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"⚠️ Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def run(self, csv_path: str = "data/sample.csv"):
        """Run the CrewAI pipeline"""
        
        agents = []
        for a_config in self.agents_config:
            
            # 👇 Dynamic Tool Loading: Reads the tool list from YAML
            yaml_tool_names = a_config.get("tools", [])
            agent_tools = []
            for tool_name in yaml_tool_names:
                if tool_name in self.available_tools:
                    agent_tools.append(self.available_tools[tool_name])
                else:
                    # Failsafe for undefined tools
                    print(f"⚠️ Warning: Tool '{tool_name}' listed for agent '{a_config['name']}' is not available.")
            
            # Agent instantiation uses the dynamically built tool list
            agents.append(
                Agent(
                    role=a_config["role"],
                    goal=a_config["goal"],
                    backstory=a_config["backstory"],
                    tools=agent_tools, # <-- Tools are now loaded from YAML and mapped
                    verbose=True,
                    allow_delegation=a_config.get("allow_delegation", False),
                    llm=self.llm,
                )
            )

        # Tasks instantiation remains the same, but agent parameter is matched by index
        tasks = []
        for t_config, agent in zip(self.tasks_config, agents):
            # Replace placeholder in description
            desc = t_config["description"].replace("{csv_path}", csv_path) 
            tasks.append(
                Task(
                    description=desc,
                    expected_output=t_config["expected_output"],
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
        
        return result
