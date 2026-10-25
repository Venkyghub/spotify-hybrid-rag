
import pytest
from backend.knowledge_graph.entity_extractor import extract_entities

from backend.knowledge_graph.graph_schema import (
    validate_node_label,
    validate_relationship_type,
    validate_node_properties,
)


def test_valid_node_label():
    assert validate_node_label("Technology") == "Technology"


def test_rejects_unknown_node_label():
    with pytest.raises(ValueError):
        validate_node_label("UnknownNode")


def test_valid_relationship_type():
    assert validate_relationship_type("USES") == "USES"


def test_rejects_unknown_relationship_type():
    with pytest.raises(ValueError):
        validate_relationship_type("UNKNOWN_RELATION")


def test_rejects_unapproved_node_property():
    with pytest.raises(ValueError):
        validate_node_properties(
            "Technology",
            {"name": "Next.js", "password": "unexpected"},
        )


def test_document_chunk_requires_citation_metadata():
    with pytest.raises(ValueError):
        validate_node_properties(
            "DocumentChunk",
            {"chunk_id": 1, "text": "Architecture text"},
        )

def test_extracts_known_technology_and_source_metadata():
    chunks = [{
        "source": "spotify_web_app_architecture.pdf",
        "page": 1,
        "chunk_id": 0,
        "text": "Frontend uses Next.js with React and TypeScript.",
    }]

    entities = extract_entities(chunks)
    names = {entity["name"] for entity in entities}

    assert {"Next.js", "React", "TypeScript"} <= names
    assert all(
        entity["source"] == "spotify_web_app_architecture.pdf"
        and entity["page"] == 1
        and entity["chunk_id"] == 0
        for entity in entities
    )


def test_normalizes_known_alias():
    chunks = [{
        "source": "architecture.pdf",
        "page": 2,
        "chunk_id": 1,
        "text": "React Query is also known as TanStack Query.",
    }]

    entities = extract_entities(chunks)

    assert sum(e["name"] == "React Query" for e in entities) == 1
    assert entities[0]["name"] == "React Query"


def test_ignores_unknown_terms():
    chunks = [{
        "source": "architecture.pdf",
        "page": 1,
        "chunk_id": 0,
        "text": "A fictional framework called UnknownStack is mentioned.",
    }]

    assert extract_entities(chunks) == []
