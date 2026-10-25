
import os

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError


class Neo4jClient:
    """Manage the Neo4j connection and execute parameterized queries."""

    def __init__(self):
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        username = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD")

        if not password:
            raise RuntimeError("NEO4J_PASSWORD is not configured.")

        self.driver = GraphDatabase.driver(
            uri,
            auth=(username, password),
        )

    def verify_connection(self) -> bool:
        """Check that Neo4j is reachable and authenticated."""
        self.driver.verify_connectivity()
        return True

    def execute_query(self, query: str, parameters: dict | None = None):
        """Execute a parameterized Cypher query and return its records."""
        try:
            records, summary, keys = self.driver.execute_query(
                query,
                parameters_=parameters or {},
            )
            return [record.data() for record in records]
        except Neo4jError:
            raise

    def close(self) -> None:
        """Release the Neo4j driver resources."""
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
