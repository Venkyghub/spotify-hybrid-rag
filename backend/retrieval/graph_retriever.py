
import re

from backend.knowledge_graph.neo4j_client import Neo4jClient


def _extract_terms(question: str) -> list[str]:
    """Extract meaningful terms from a question."""
    if not isinstance(question, str) or not question.strip():
        return []

    stop_words = {
        "a", "an", "and", "are", "by", "can", "could", "do", "does",
        "for", "how", "i", "in", "is", "it", "me", "of", "on",
        "or", "please", "the", "their", "them", "this", "to",
        "use", "used", "using", "what", "which", "with", "would",
        "support", "supports", "supporting",
        "technology", "technologies",
    }

    words = re.findall(
        r"[a-zA-Z0-9]+(?:[.&/-][a-zA-Z0-9]+)*",
        question,
    )

    return [
        word for word in dict.fromkeys(words)
        if word.lower() not in stop_words and len(word) > 1
    ]


def retrieve_graph_context(
    client: Neo4jClient,
    question: str,
    limit: int = 10,
) -> list[dict]:
    """Retrieve relevant graph relationships with unique evidence chunks."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not isinstance(limit, int) or not 1 <= limit <= 50:
        raise ValueError("limit must be between 1 and 50.")

    terms = _extract_terms(question)
    if not terms:
        return []

    query = """
    MATCH (e)-[r:USES|SUPPORTS]->(related)
    WHERE e.name IS NOT NULL
      AND related.name IS NOT NULL
      AND (
          any(term IN $terms
              WHERE toLower(e.name) CONTAINS toLower(term))
          OR any(term IN $terms
              WHERE toLower(related.name) CONTAINS toLower(term))
      )
      AND NOT e:DocumentChunk
      AND NOT e:ArchitectureDocument
      AND NOT related:DocumentChunk
      AND NOT related:ArchitectureDocument

    MATCH (c:DocumentChunk)-[:MENTIONS]->(e)
    WHERE toLower(coalesce(c.text, ''))
          CONTAINS toLower(related.name)

    WITH DISTINCT e, r, related, c,
         size([
             term IN $terms
             WHERE toLower(coalesce(c.text, ''))
                   CONTAINS toLower(term)
         ]) AS term_score

    RETURN
        labels(e) AS entity_labels,
        e.name AS entity,
        type(r) AS relationship,
        related.name AS related_entity,
        c.source AS source,
        c.page AS page,
        c.chunk_id AS chunk_id,
        c.text AS evidence,
        term_score

    ORDER BY term_score DESC, c.page ASC, c.chunk_id ASC
    LIMIT $limit
    """

    records = client.execute_query(
        query,
        {"terms": terms, "question": question, "limit": limit},
    )

    # Also retrieve document chunks linked directly to graph evidence entities.
    evidence_query = """
    MATCH (c:DocumentChunk)-[:SUPPORTS_EVIDENCE]->(e)
    WHERE e.name IS NOT NULL
      AND any(term IN $terms
              WHERE toLower(coalesce(e.name, '')) CONTAINS toLower(term))
    WITH c, e,
         size([
             term IN $terms
             WHERE toLower(coalesce(c.text, ''))
                   CONTAINS toLower(term)
         ]) AS term_score
    RETURN
        labels(e) AS entity_labels,
        e.name AS entity,
        'SUPPORTS_EVIDENCE' AS relationship,
        e.name AS related_entity,
        c.source AS source,
        c.page AS page,
        c.chunk_id AS chunk_id,
        c.text AS evidence,
        term_score
    ORDER BY term_score DESC, c.page ASC, c.chunk_id ASC
    LIMIT $limit
    """

    evidence_records = client.execute_query(
        evidence_query,
        {"terms": terms, "limit": limit},
    )
    records = list(records or []) + list(evidence_records or [])

    # Fallback: retrieve relevant chunks through actual MENTIONS relationships.
    if not records:
        fallback_query = """
        MATCH (c:DocumentChunk)-[:MENTIONS]->(e)
        WHERE e.name IS NOT NULL
          AND NOT e:DocumentChunk
          AND NOT e:ArchitectureDocument
        WITH c,
             collect(DISTINCT e.name) AS entities,
             size([
                 term IN $terms
                 WHERE toLower(coalesce(c.text, ''))
                       CONTAINS toLower(term)
             ]) AS term_score
        WHERE term_score > 0
        RETURN
            labels(c) AS entity_labels,
            'DocumentChunk' AS entity,
            'MENTIONS' AS relationship,
            entities AS related_entity,
            c.source AS source,
            c.page AS page,
            c.chunk_id AS chunk_id,
            c.text AS evidence,
            term_score
        ORDER BY term_score DESC, c.page ASC, c.chunk_id ASC
        LIMIT $limit
        """
        records = client.execute_query(
            fallback_query,
            {"terms": terms, "limit": limit},
        )

    results = []
    seen = set()

    for record in records:
        result = {
            "entity_labels": record.get("entity_labels", []),
            "entity": record.get("entity"),
            "relationship": record.get("relationship"),
            "related_entity": record.get("related_entity"),
            "source": record.get("source"),
            "page": record.get("page"),
            "chunk_id": record.get("chunk_id"),
            "evidence": record.get("evidence"),
            "query_term_score": record.get("term_score", 0),
        }

        key = (
            result["source"],
            result["chunk_id"],
        )

        if key not in seen:
            seen.add(key)
            results.append(result)

    return results
