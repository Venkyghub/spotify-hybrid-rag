
import re

from backend.knowledge_graph.neo4j_client import Neo4jClient


def _extract_terms(question: str) -> list[str]:
    """Extract useful query terms while excluding common stop words."""

    if not isinstance(question, str) or not question.strip():
        return []

    stop_words = {
        "a", "an", "and", "are", "can", "could", "do", "does",
        "for", "how", "i", "in", "is", "it", "me", "of", "on",
        "or", "please", "the", "their", "them", "this", "to",
        "use", "used", "using", "what", "which", "with", "would",
        "support", "supports", "supporting",
    }

    words = re.findall(
        r"[a-zA-Z0-9]+(?:[.&/-][a-zA-Z0-9]+)*",
        question,
    )

    return [
        word
        for word in dict.fromkeys(words)
        if word.lower() not in stop_words and len(word) > 1
    ]


def retrieve_graph_context(
    client: Neo4jClient,
    question: str,
    limit: int = 10,
) -> list[dict]:
    """
    Retrieve relevant graph entities, relationships, and evidence.

    Evidence chunks are ranked by whether they mention both endpoints
    of a relationship and whether they contain terms from the question.

    Important:
        Evidence ranking improves relevance but does not independently
        prove that a chunk supports a particular graph relationship.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not isinstance(limit, int) or not 1 <= limit <= 50:
        raise ValueError("limit must be between 1 and 50.")

    terms = _extract_terms(question)

    if not terms:
        return []

    query = """
    MATCH (e)
    WHERE e.name IS NOT NULL
      AND NOT e:DocumentChunk
      AND NOT e:ArchitectureDocument
      AND any(
          term IN $terms
          WHERE toLower(e.name) CONTAINS toLower(term)
      )

    WITH DISTINCT e
    LIMIT $limit

    OPTIONAL MATCH (e)-[r:USES|SUPPORTS]-(related)
    WHERE related.name IS NOT NULL
      AND NOT related:DocumentChunk
      AND NOT related:ArchitectureDocument

    WITH e, r, related

    OPTIONAL MATCH (c:DocumentChunk)-[:MENTIONS]->(e)
    WHERE related IS NULL
        OR toLower(coalesce(c.text, ''))
           CONTAINS toLower(related.name)

    WITH e, r, related, c,
         CASE
             WHEN c IS NULL THEN 0
             ELSE
                 CASE
                     WHEN related IS NOT NULL
                      AND toLower(coalesce(c.text, ''))
                          CONTAINS toLower(e.name)
                      AND toLower(coalesce(c.text, ''))
                          CONTAINS toLower(related.name)
                     THEN 3
                     WHEN toLower(coalesce(c.text, ''))
                          CONTAINS toLower(e.name)
                     THEN 2
                     ELSE 1
                 END
         END AS endpoint_score,
         size([
             term IN $terms
             WHERE toLower(coalesce(c.text, ''))
                   CONTAINS toLower(term)
         ]) AS query_term_score

    WITH e, r, related, c, endpoint_score, query_term_score
    ORDER BY
        endpoint_score DESC,
        query_term_score DESC,
        c.chunk_id ASC

    RETURN
        labels(e) AS entity_labels,
        e.name AS entity,
        type(r) AS relationship,
        related.name AS related_entity,
        c.source AS source,
        c.page AS page,
        c.chunk_id AS chunk_id,
        c.text AS evidence,
        endpoint_score,
        query_term_score

    LIMIT $limit
    """

    records = client.execute_query(
        query,
        {
            "terms": terms,
            "limit": limit,
        },
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
            "endpoint_score": record.get("endpoint_score", 0),
            "query_term_score": record.get("query_term_score", 0),
        }

        key = (
            result["entity"],
            result["relationship"],
            result["related_entity"],
            result["source"],
            result["chunk_id"],
        )

        if key not in seen:
            seen.add(key)
            results.append(result)

    return results
