
"""Schema definitions for the Spotify architecture knowledge graph."""

# Allowed node labels.
NODE_LABELS = {
    "Component",
    "Technology",
    "Feature",
    "Service",
    "DataStore",
    "Protocol",
    "ArchitectureDocument",
    "DocumentChunk",
}

# Allowed relationship types.
RELATIONSHIP_TYPES = {
    "USES",
    "CONNECTS_TO",
    "DEPENDS_ON",
    "IMPLEMENTS",
    "STORES",
    "AUTHENTICATES_WITH",
    "SUPPORTS",
    "PART_OF",
    "DOCUMENTED_IN",
    "MENTIONS",
    "SUPPORTS_EVIDENCE",
}

# Properties permitted for nodes, grouped by label.
NODE_PROPERTIES = {
    "Component": {"name", "description"},
    "Technology": {"name", "category", "description"},
    "Feature": {"name", "description"},
    "Service": {"name", "description"},
    "DataStore": {"name", "store_type", "description"},
    "Protocol": {"name", "description"},
    "ArchitectureDocument": {"source"},
    "DocumentChunk": {"chunk_id", "source", "page", "text"},
}


def validate_node_label(label: str) -> str:
    """Reject labels outside the approved schema."""
    if label not in NODE_LABELS:
        raise ValueError(f"Unsupported node label: {label}")
    return label


def validate_relationship_type(relationship_type: str) -> str:
    """Reject relationship types outside the approved schema."""
    if relationship_type not in RELATIONSHIP_TYPES:
        raise ValueError(
            f"Unsupported relationship type: {relationship_type}"
        )
    return relationship_type


def validate_node_properties(label: str, properties: dict) -> dict:
    """Allow only approved properties for a node label."""
    validate_node_label(label)

    allowed = NODE_PROPERTIES[label]
    unexpected = set(properties) - allowed

    if unexpected:
        raise ValueError(
            f"Unsupported properties for {label}: {sorted(unexpected)}"
        )

    if label not in {"ArchitectureDocument"} and not properties.get("name") \
            and label != "DocumentChunk":
        raise ValueError(f"{label} nodes must have a non-empty name.")

    if label == "DocumentChunk":
        if "chunk_id" not in properties:
            raise ValueError("DocumentChunk nodes require chunk_id.")
        if "source" not in properties or "page" not in properties:
            raise ValueError(
                "DocumentChunk nodes require source and page metadata."
            )

    return properties
