
import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from backend.knowledge_graph.neo4j_client import Neo4jClient
from backend.knowledge_graph.graph_builder import build_graph


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_store_chunks_in_neo4j():
    load_dotenv(PROJECT_ROOT / ".env")

    if not os.getenv("NEO4J_PASSWORD"):
        pytest.skip("NEO4J_PASSWORD is not configured.")

    with (PROJECT_ROOT / "data" / "processed" / "chunks.json").open(
        "r", encoding="utf-8"
    ) as file:
        chunks = json.load(file)

    with Neo4jClient() as client:
        assert client.verify_connection()
        count = build_graph(client, chunks)

        assert count == len(chunks)

        result = client.execute_query(
            """
            MATCH (c:DocumentChunk)
            WHERE c.source = $source
            RETURN count(c) AS total
            """,
            {"source": chunks[0]["source"]},
        )

        assert result[0]["total"] == len(chunks)

def test_store_chunks_in_neo4j():
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")

    if not os.getenv("NEO4J_PASSWORD"):
        pytest.skip("NEO4J_PASSWORD is not configured.")

    chunks_path = project_root / "data" / "processed" / "chunks.json"

    with chunks_path.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    with Neo4jClient() as client:
        assert client.verify_connection()

        count = build_graph(client, chunks)
        assert count == len(chunks)

        result = client.execute_query(
            """
            MATCH (c:DocumentChunk)
            WHERE c.source = $source
            RETURN count(c) AS total
            """,
            {"source": chunks[0]["source"]}
        )

        assert result[0]["total"] == len(chunks)
