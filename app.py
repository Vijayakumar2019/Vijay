import warnings
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
import sys
from unittest.mock import MagicMock

# 1. THE "DLL SURGERY": Prevent Torch from loading locally to avoid WinError 1114
# This must happen before ANY other imports.
sys.modules["torch"] = MagicMock()
sys.modules["transformers"] = MagicMock()



import streamlit as st
import io
import os
from docx import Document
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Import your graph engine (ensure graph_engine.py is in the same folder)
from graph_engine import app_graph

# --- UI CONFIGURATION ---
st.set_page_config(page_title="Agentic Resume Skill-Graph", page_icon="🚀", layout="wide")

st.title("🚀 Agentic Resume Skill-Graph")
st.markdown("""
Extracts skills from resumes using **Azure OpenAI** and maps them into a **Neo4j Knowledge Graph**.
""")

# --- SIDEBAR / SETTINGS ---
with st.sidebar:
    st.header("System Status")
    if os.getenv("AZURE_OPENAI_API_KEY"):
        st.success("Azure OpenAI: Connected")
    else:
        st.error("Azure OpenAI: Key Missing")
        
    if os.getenv("NEO4J_URI"):
        st.success("Neo4j Database: Configured")
    else:
        st.error("Neo4j: Credentials Missing")

# --- FILE UPLOAD SECTION ---
uploaded_file = st.file_uploader("Upload Resume", type=["docx", "txt"])

if uploaded_file is not None:
    st.info(f"Processing: {uploaded_file.name}")
    
    try:
        # 2. ROBUST TEXT EXTRACTION
        if uploaded_file.name.endswith(".docx"):
            # Use python-docx to parse binary Word files
            doc_file = io.BytesIO(uploaded_file.getvalue())
            doc = Document(doc_file)
            raw_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        else:
            # Standard UTF-8 decoding for text files
            raw_text = uploaded_file.getvalue().decode("utf-8", errors="ignore")

        # --- EXECUTION BUTTON ---
        if st.button("Analyze & Map to Graph"):
            with st.spinner("🤖 Agents are collaborating on extraction and graph mapping..."):
                # 3. KICK OFF THE LANGGRAPH ORCHESTRATOR
                # We pass the raw text into the starting state of the graph
                final_state = app_graph.invoke({"raw_text": raw_text})
                
                # --- RESULTS DISPLAY ---
                st.divider()
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📝 Extracted Entities")
                    st.json(final_state.get("extracted_data", {}))
                
                with col2:
                    st.subheader("🌐 Graph Status")
                    status = final_state.get("graph_status", "Unknown")
                    if "Success" in status:
                        st.success("Successfully injected into Neo4j!")
                    else:
                        st.warning(f"Status: {status}")
                
                st.subheader("📄 Extracted Text Preview")
                st.text_area("Content", raw_text[:1000] + "...", height=200)

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
        
        

# --- FOOTER ---
#st.divider()
import streamlit as st
from graph_engine import search_graph

st.divider()
st.subheader("🔍 Query your Talent Map")
user_input = st.text_input("Ask a question (e.g., 'Find candidates with Azure skills')")

if user_input:
    with st.spinner("Thinking..."):
        answer = search_graph(user_input)
        st.write(f"**Answer:** {answer}")
st.caption("Powered by LangGraph, Neo4j, and Azure OpenAI | 2026 AI Architect Suite")