import sys
from unittest.mock import MagicMock

# Force Python to think these libraries are missing so it uses 'tiktoken' fallback
sys.modules["torch"] = MagicMock()
sys.modules["transformers"] = MagicMock()

# Now import your actual tools
import streamlit as st
from langchain_openai import AzureChatOpenAI
import os
import platform
import ctypes
from importlib.util import find_spec

# MUST BE THE FIRST BLOCK OF CODE
if platform.system() == "Windows":
    try:
        # Locate the problematic DLL inside your .venv
        if (spec := find_spec("torch")) and spec.origin:
            torch_lib_path = os.path.join(os.path.dirname(spec.origin), "lib", "c10.dll")
            if os.path.exists(torch_lib_path):
                # Force the DLL to initialize manually
                ctypes.CDLL(os.path.normpath(torch_lib_path))
    except Exception as e:
        print(f"DLL Pre-load failed: {e}")

import os
from typing import List, TypedDict
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# LangChain & LangGraph Imports
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, START, END

# Import your database manager
from db_utils import Neo4jManager

load_dotenv()

# --- 1. DEFINE THE STRUCTURED OUTPUT SCHEMA ---
class CandidateSchema(BaseModel):
    """Information extracted from a resume for knowledge graph ingestion."""
    name: str = Field(description="The full name of the candidate")
    skills: List[str] = Field(description="A list of technical skills, programming languages, or tools")
    company: str = Field(description="The most recent or current company the candidate works for")

# --- 2. DEFINE THE GRAPH STATE ---
class AgentState(TypedDict):
    raw_text: str            # The full resume text from Streamlit
    extracted_data: dict     # The JSON result from Agent 1
    graph_status: str        # Status message for the UI

# --- 3. INITIALIZE THE LLM ---
llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0  # Set to 0 for consistent extraction
)

# Bind the schema to the LLM (Native Structured Output)
structured_llm = llm.with_structured_output(CandidateSchema)

# --- 4. DEFINE THE NODES ---

def extraction_node(state: AgentState):
    """Agent 1: Uses Azure OpenAI with a strict JSON method to avoid warnings."""
    # Force the use of json_schema method to prevent serialization warnings
    structured_llm = llm.with_structured_output(CandidateSchema, method="json_schema")
    
    prompt = f"Analyze the following resume and extract the candidate details:\n\n{state['raw_text']}"
    
    # Invoke the model
    result = structured_llm.invoke(prompt)
    
    # Return as a dictionary
    return {
        "extracted_data": result.model_dump(), 
        "graph_status": "Extraction Complete"
    }

def graph_ingest_node(state: AgentState):
    """Agent 2: Ingests the structured JSON into the Neo4j Knowledge Graph."""
    db = Neo4jManager()
    data = state['extracted_data']
    
    try:
        # Cypher query to create the candidate-skill relationship
        query = """
        MERGE (c:Candidate {name: $name})
        SET c.company = $company
        WITH c
        UNWIND $skills AS skill_name
        MERGE (s:Skill {name: skill_name})
        MERGE (c)-[:HAS_SKILL]->(s)
        """
        db.execute_query(query, data)
        return {"graph_status": f"Success: {data['name']} mapped to Neo4j"}
    except Exception as e:
        return {"graph_status": f"Graph Error: {str(e)}"}
    finally:
        db.close()
        
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph

def search_graph(user_query: str):
    # 1. Connect to the existing graph
    graph = Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USER"),
        password=os.getenv("NEO4J_PW")
    )
    
    # 2. Initialize the translation chain
    # We set verbose=True so you can see the Cypher it generates in the console!
    chain = GraphCypherQAChain.from_llm(
        llm=llm, 
        graph=graph, 
        verbose=True,
        allow_dangerous_requests=True # Required in 2026 for security safety
    )
    
    # 3. Ask the question
    response = chain.invoke({"query": user_query})
    return response["result"]

# --- 5. ASSEMBLE THE GRAPH ---
workflow = StateGraph(AgentState)

# Add our two agents as nodes
workflow.add_node("extractor", extraction_node)
workflow.add_node("ingestor", graph_ingest_node)

# Define the workflow sequence
workflow.add_edge(START, "extractor")
workflow.add_edge("extractor", "ingestor")
workflow.add_edge("ingestor", END)

# Compile the graph into an executable app
app_graph = workflow.compile()