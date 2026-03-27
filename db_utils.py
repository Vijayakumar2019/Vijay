from neo4j import GraphDatabase
import os

class Neo4jManager:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"), 
            auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PW"))
        )

    def execute_query(self, query, params=None):
        with self.driver.session() as session:
            return session.run(query, params or {}).data()

    def close(self):
        self.driver.close()