import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
import httpx

load_dotenv()

def test_connections():
    # 1. Test Azure OpenAI Endpoint
    print("Testing Azure OpenAI...")
    try:
        url = os.getenv("AZURE_OPENAI_ENDPOINT")
        response = httpx.get(url, timeout=5.0)
        print(f"✅ Azure Reachable (Status: {response.status_code})")
    except Exception as e:
        print(f"❌ Azure Connection Failed: {e}")

    # 2. Test Neo4j
    print("\nTesting Neo4j...")
    try:
        driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"), 
            auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PW"))
        )
        driver.verify_connectivity()
        print("✅ Neo4j Reachable")
        driver.close()
    except Exception as e:
        print(f"❌ Neo4j Connection Failed: {e}")

if __name__ == "__main__":
    test_connections()