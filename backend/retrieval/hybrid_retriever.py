
from backend.knowledge_graph.neo4j_client import Neo4jClient
from backend.retrieval.graph_retriever import retrieve_graph_context
from backend.retrieval.vector_retriever import retrieve_vector_context
from backend.retrieval.reranker import rerank_results

def retrieve_hybrid_context(
    question: str,
    graph_client: Neo4jClient,
    top_k: int = 5,
    graph_limit: int = 10,
) -> list[dict]:
    """
    Combine FAISS vector retrieval and Neo4j graph retrieval.

    Behavior:
    1. Retrieve semantically relevant chunks from FAISS.
    2. Retrieve entities and relationships from Neo4j.
    3. Merge results that refer to the same source chunk.
    4. Preserve graph relationship metadata on matching vector results.
    5. Retain graph-only results when they identify distinct entities
       or relationships.
    6. Avoid unnecessary duplicate results.

    Note:
        This function combines retrieval results. It does not itself
        verify whether every graph relationship is supported by exact
        relationship-level evidence.
    """

    # ---------------------------------------------------------
    # 1. Validate inputs
    # ---------------------------------------------------------
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not isinstance(top_k, int) or not 1 <= top_k <= 50:
        raise ValueError("top_k must be between 1 and 50.")

    if not isinstance(graph_limit, int) or not 1 <= graph_limit <= 50:
        raise ValueError("graph_limit must be between 1 and 50.")

    # ---------------------------------------------------------
    # 2. Retrieve vector results from FAISS
    # ---------------------------------------------------------
    vector_results = retrieve_vector_context(
        question=question.strip(),
        top_k=top_k,
    )

    # ---------------------------------------------------------
    # 3. Retrieve graph results from Neo4j
    # ---------------------------------------------------------
    graph_results = retrieve_graph_context(
        client=graph_client,
        question=question.strip(),
        limit=graph_limit,
    )

    # ---------------------------------------------------------
    # 4. Initialize merged results and deduplication tracking
    # ---------------------------------------------------------
    merged: list[dict] = []

    # Maps a source-chunk key to its position in merged results.
    chunk_positions: dict[tuple, int] = {}

    # Tracks graph-only relationships to prevent duplicate records.
    graph_relationship_keys: set[tuple] = set()

    # ---------------------------------------------------------
    # 5. Add vector results first
    # ---------------------------------------------------------
    for item in vector_results:
        result = {
            **item,
            "retrieval_source": "vector",
            "graph_relationships": [],
        }

        source = result.get("source")
        chunk_id = result.get("chunk_id")

        if source is not None and chunk_id is not None:
            chunk_key = (source, chunk_id)

            if chunk_key in chunk_positions:
                # Avoid adding the same source chunk twice.
                existing = merged[chunk_positions[chunk_key]]

                if result.get("similarity") is not None:
                    existing_similarity = existing.get("similarity")

                    if (
                        existing_similarity is None
                        or result["similarity"] > existing_similarity
                    ):
                        existing["similarity"] = result["similarity"]

                continue

            chunk_positions[chunk_key] = len(merged)

        merged.append(result)

    # ---------------------------------------------------------
    # 6. Merge graph results
    # ---------------------------------------------------------
    for item in graph_results:
        result = {
            **item,
            "retrieval_source": "graph",
        }

        source = result.get("source")
        chunk_id = result.get("chunk_id")

        entity = result.get("entity")
        relationship = result.get("relationship")
        related_entity = result.get("related_entity")

        # Build a stable key for the graph relationship.
        relationship_key = (
            entity,
            relationship,
            related_entity,
        )

        # Identify graph results referring to a source chunk.
        chunk_key = (
            (source, chunk_id)
            if source is not None and chunk_id is not None
            else None
        )

        relationship_info = {
            "entity": entity,
            "relationship": relationship,
            "related_entity": related_entity,
            "entity_labels": result.get("entity_labels"),
            "source": source,
            "page": result.get("page"),
            "chunk_id": chunk_id,
            "evidence": result.get("evidence"),
        }

        # -----------------------------------------------------
        # 6A. Matching vector chunk: enrich it with graph data.
        # -----------------------------------------------------
        if (
            chunk_key is not None
            and chunk_key in chunk_positions
        ):
            position = chunk_positions[chunk_key]
            existing = merged[position]

            existing["retrieval_source"] = "hybrid"

            existing.setdefault("graph_relationships", [])

            existing_relationships = existing["graph_relationships"]

            if relationship_info not in existing_relationships:
                existing_relationships.append(relationship_info)

            continue

        # -----------------------------------------------------
        # 6B. Graph-only result: retain distinct relationships.
        # -----------------------------------------------------
        if relationship_key in graph_relationship_keys:
            # Same relationship may have evidence from another chunk.
            # Keep distinct evidence records where available.
            for existing in merged:
                existing_relationships = existing.get(
                    "graph_relationships", []
                )

                matching_relationship = any(
                    (
                        rel.get("entity"),
                        rel.get("relationship"),
                        rel.get("related_entity"),
                    ) == relationship_key
                    for rel in existing_relationships
                )

                if matching_relationship:
                    if relationship_info not in existing_relationships:
                        existing_relationships.append(relationship_info)
                    break

            continue

        graph_relationship_keys.add(relationship_key)

        # Avoid adding a graph-only record without useful identifying data.
        if not any(
            value is not None
            for value in (entity, relationship, related_entity)
        ):
            continue

        result["graph_relationships"] = [relationship_info]
        merged.append(result)

    return rerank_results(
        question=question.strip(),
        results=merged,
        top_k=top_k,
    )
