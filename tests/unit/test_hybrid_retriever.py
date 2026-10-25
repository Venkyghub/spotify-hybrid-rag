from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

from backend.knowledge_graph.neo4j_client import Neo4jClient
from backend.retrieval.hybrid_retriever import retrieve_hybrid_context


def main():
    client = Neo4jClient()

    try:
        question = (
            "Which search technologies support full-text "
            "indexing and autocomplete?"
        )

        results = retrieve_hybrid_context(
            question,
            client,
            top_k=5,
            graph_limit=10,
        )

        print(f"\nTOTAL RESULTS: {len(results)}\n")

        for i, result in enumerate(results, start=1):
            print(f"Result {i}")
            print("Source:", result.get("retrieval_source"))
            print("Page:", result.get("page"))
            print("Chunk:", result.get("chunk_id"))
            print("Entity:", result.get("entity"))
            print("Relationship:", result.get("relationship"))
            print("Related entity:", result.get("related_entity"))
            print("Similarity:", result.get("similarity"))
            print("Graph relationships:", result.get("graph_relationships", []))
            print("-" * 50)

    finally:
        client.close()


if __name__ == "__main__":
    main()
