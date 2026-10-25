
import re

from backend.knowledge_graph.graph_schema import validate_relationship_type


RELATIONSHIP_RULES = [
    {
        "source": "Auth & Identity Service",
        "target": "JWT",
        "relationship": "SUPPORTS",
        "patterns": [
            r"\bJWT\b",
            r"\btoken\b",
        ],
        "evidence_patterns": [
            r"(?:JWT|token).{0,100}(?:auth|identity|login|session|issuance|refresh)",
            r"(?:auth|identity|login|session|issuance|refresh).{0,100}(?:JWT|token)",
        ],
    },
    {
        "source": "Auth & Identity Service",
        "target": "OAuth2",
        "relationship": "SUPPORTS",
        "patterns": [
            r"\bOAuth2\b",
        ],
        "evidence_patterns": [
            r"\bOAuth2\b",
        ],
    },
    {
        "source": "Auth & Identity Service",
        "target": "OIDC",
        "relationship": "SUPPORTS",
        "patterns": [
            r"\bOIDC\b",
            r"\bOpenID Connect\b",
        ],
        "evidence_patterns": [
            r"\bOIDC\b",
            r"\bOpenID Connect\b",
        ],
    },
    {
        "source": "Search & Discovery Index Service",
        "target": "Elasticsearch",
        "relationship": "USES",
        "patterns": [
            r"\bElasticsearch\s*/\s*OpenSearch\s*:",
            r"\bElasticsearch\s*/\s*OpenSearch\b",
        ],
        "evidence_patterns": [
            r"\bElasticsearch\s*/\s*OpenSearch\b",
        ],
    },
    {
        "source": "Search & Discovery Index Service",
        "target": "OpenSearch",
        "relationship": "USES",
        "patterns": [
            r"\bElasticsearch\s*/\s*OpenSearch\s*:",
            r"\bElasticsearch\s*/\s*OpenSearch\b",
        ],
        "evidence_patterns": [
            r"\bElasticsearch\s*/\s*OpenSearch\b",
        ],
    },
]



def extract_relationships(chunks: list[dict]) -> list[dict]:
    """
    Extract rule-based relationships and aggregate evidence
    from multiple chunks for the same entity relationship.
    """
    relationships_by_key = {}

    for chunk in chunks:
        text = chunk.get("text", "")

        if not isinstance(text, str) or not text.strip():
            continue

        for rule in RELATIONSHIP_RULES:
            trigger_match = any(
                re.search(pattern, text, re.IGNORECASE)
                for pattern in rule["patterns"]
            )
            evidence_match = any(
                re.search(pattern, text, re.IGNORECASE)
                for pattern in rule["evidence_patterns"]
            )

            if not trigger_match or not evidence_match:
                continue

            relationship_type = rule["relationship"]

            if not validate_relationship_type(relationship_type):
                raise ValueError(
                    f"Unsupported relationship type: {relationship_type}"
                )

            key = (
                rule["source"],
                relationship_type,
                rule["target"],
            )

            if key not in relationships_by_key:
                relationships_by_key[key] = {
                    "source_entity": rule["source"],
                    "relationship": relationship_type,
                    "target_entity": rule["target"],
                    "evidence_items": [],
                }

            evidence_item = {
                "source_document": chunk.get("source"),
                "page": chunk.get("page"),
                "chunk_id": chunk.get("chunk_id"),
                "evidence": text[:500],
            }

            if evidence_item not in relationships_by_key[key]["evidence_items"]:
                relationships_by_key[key]["evidence_items"].append(
                    evidence_item
                )

    return list(relationships_by_key.values())
