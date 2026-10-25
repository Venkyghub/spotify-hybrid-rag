
from backend.knowledge_graph.neo4j_client import Neo4jClient
from backend.knowledge_graph.entity_extractor import extract_entities
from backend.knowledge_graph.relationship_extractor import extract_relationships
from backend.knowledge_graph.graph_schema import (
    validate_node_label,
    validate_relationship_type,
)


def build_graph(client: Neo4jClient, chunks: list[dict]) -> int:
    """
    Store document chunks, extracted entities, and supported relationships.

    Returns the number of document chunks processed.
    Relationship evidence is retained through source chunk references.
    """
    if not chunks:
        raise ValueError("Cannot build a graph without chunks.")

    sources = {item.get("source") for item in chunks}

    if len(sources) != 1 or None in sources:
        raise ValueError("All chunks must have the same source filename.")

    for item in chunks:
        if (
            item.get("chunk_id") is None
            or item.get("page") is None
            or not isinstance(item.get("text"), str)
            or not item["text"].strip()
        ):
            raise ValueError(
                "Each chunk needs an ID, page, and non-empty text."
            )

    source = next(iter(sources))

    # 1. Store the source document and its chunks.
    chunk_query = """
    MERGE (d:ArchitectureDocument {source: $source})
    WITH d
    UNWIND $chunks AS item
    MERGE (c:DocumentChunk {
        source: item.source,
        chunk_id: item.chunk_id
    })
    SET c.page = item.page,
        c.text = item.text
    MERGE (d)-[:DOCUMENTED_IN]->(c)
    RETURN count(c) AS chunks_processed
    """

    records = client.execute_query(
        chunk_query,
        {"source": source, "chunks": chunks},
    )

    chunks_processed = (
        records[0]["chunks_processed"] if records else 0
    )

    # 2. Extract and store architecture entities.
    entities = extract_entities(chunks)

    for entity in entities:
        label = entity["label"]

        if not validate_node_label(label):
            raise ValueError(f"Unsupported entity label: {label}")

        # Labels are validated against the fixed graph schema before
        # being interpolated into the Cypher query.
        entity_query = f"""
        MATCH (c:DocumentChunk {{
            source: $source,
            chunk_id: $chunk_id
        }})
        MERGE (e:{label} {{name: $name}})
        SET e.category = $category
        MERGE (c)-[:MENTIONS]->(e)
        """

        # MENTIONS must be included in your schema's allowed
        # relationship types before this step can run.
        if not validate_relationship_type("MENTIONS"):
            raise ValueError(
                "Add MENTIONS to ALLOWED_RELATIONSHIPS in graph_schema.py "
                "before storing entity mentions."
            )

        client.execute_query(
            entity_query,
            {
                "source": entity.get("source") or source,
                "chunk_id": entity["chunk_id"],
                "name": entity["name"],
                "category": entity.get("category", "Unknown"),
            },
        )

    # 3. Extract and store deduplicated architecture relationships.
    relationships = extract_relationships(chunks)

    for relationship in relationships:
        rel_type = relationship["relationship"]

        if not validate_relationship_type(rel_type):
            raise ValueError(
                f"Unsupported relationship type: {rel_type}"
            )

        # Entity names are fixed rule-catalog values; parameters carry
        # all document-derived values.
        for evidence in relationship["evidence_items"]:
            relationship_query = f"""
            MATCH (source_entity:{_entity_label(
                relationship["source_entity"]
            )} {{name: $source_name}})
            MATCH (target_entity:{_entity_label(
                relationship["target_entity"]
            )} {{name: $target_name}})
            MERGE (source_entity)-[r:{rel_type}]->(target_entity)
            SET r.source_document = $source_document,
                r.last_evidence_page = $page
            WITH source_entity, target_entity
            MATCH (c:DocumentChunk {{
                source: $source_document,
                chunk_id: $chunk_id
            }})
            MERGE (c)-[:SUPPORTS_EVIDENCE]->(source_entity)
            MERGE (c)-[:SUPPORTS_EVIDENCE]->(target_entity)
            """

            client.execute_query(
                relationship_query,
                {
                    "source_name": relationship["source_entity"],
                    "target_name": relationship["target_entity"],
                    "source_document": evidence["source_document"],
                    "page": evidence["page"],
                    "chunk_id": evidence["chunk_id"],
                },
            )

    return chunks_processed


def _entity_label(entity_name: str) -> str:
    """Return a validated label for a known entity catalog entry."""
    entities = {
        entity["name"]: entity["label"]
        for entity in extract_entities([
            {
                "source": "schema-validation",
                "page": 0,
                "chunk_id": "schema-validation",
                "text": entity_text,
            }
            for entity_text in [entity_name]
        ])
    }

    # Entity extraction is text-based, so use the actual entity catalog
    # instead of deriving labels from arbitrary query text.
    from backend.knowledge_graph.entity_extractor import ENTITY_CATALOG

    for name, definition in ENTITY_CATALOG.items():
        if name == entity_name:
            label = definition["label"]
            if not validate_node_label(label):
                break
            return label

    raise ValueError(
        f"Entity '{entity_name}' is not present in the entity catalog."
    )
