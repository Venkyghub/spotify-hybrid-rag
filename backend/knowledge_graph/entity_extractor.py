
import re


# Explicit aliases for technologies and architecture concepts.
# Extend this dictionary when new terms appear in the source document.
ENTITY_CATALOG = {
    "Next.js": {
        "label": "Technology",
        "aliases": ["Next.js", "Next JS"],
        "category": "Frontend framework",
    },
    "React": {
        "label": "Technology",
        "aliases": ["React", "React.js"],
        "category": "Frontend library",
    },
    "TypeScript": {
        "label": "Technology",
        "aliases": ["TypeScript"],
        "category": "Programming language",
    },
    "Tailwind CSS": {
        "label": "Technology",
        "aliases": ["Tailwind CSS"],
        "category": "CSS framework",
    },
    "React Query": {
        "label": "Technology",
        "aliases": ["React Query", "TanStack Query"],
        "category": "Data-fetching library",
    },
    "Zustand": {
        "label": "Technology",
        "aliases": ["Zustand"],
        "category": "State management",
    },
    "Recoil": {
        "label": "Technology",
        "aliases": ["Recoil"],
        "category": "State management",
    },
    "Jotai": {
        "label": "Technology",
        "aliases": ["Jotai"],
        "category": "State management",
    },
    "HLS.js": {
        "label": "Technology",
        "aliases": ["HLS.js", "HLS JS"],
        "category": "Media streaming",
    },
    "HTML5 Audio": {
        "label": "Technology",
        "aliases": ["HTML5 Audio"],
        "category": "Audio playback",
    },
    "Web Audio API": {
        "label": "Technology",
        "aliases": ["Web Audio API"],
        "category": "Audio processing",
    },
    "OAuth2": {
        "label": "Protocol",
        "aliases": ["OAuth2", "OAuth 2.0"],
        "category": "Authentication",
    },
    "OIDC": {
        "label": "Protocol",
        "aliases": ["OIDC", "OpenID Connect"],
        "category": "Authentication",
    },

    
    # Databases and storage
    "PostgreSQL": {
        "label": "DataStore",
        "aliases": ["PostgreSQL", "Postgres"],
        "category": "Relational database",
    },
    "Redis": {
        "label": "DataStore",
        "aliases": ["Redis"],
        "category": "Cache and session store",
    },
    "Elasticsearch": {
        "label": "Technology",
        "aliases": ["Elasticsearch", "ElasticSearch"],
        "category": "Search engine",
    },
    "OpenSearch": {
        "label": "Technology",
        "aliases": ["OpenSearch"],
        "category": "Search engine",
    },
    "ClickHouse": {
        "label": "DataStore",
        "aliases": ["ClickHouse"],
        "category": "Analytics data warehouse",
    },
    "S3": {
        "label": "DataStore",
        "aliases": ["S3-compatible object storage", "S3-like", "S3"],
        "category": "Object storage",
    },
    "CDN": {
        "label": "Technology",
        "aliases": ["CDN", "CloudFront", "Fastly", "Cloudflare"],
        "category": "Content delivery network",
    },
    # Infrastructure
    "Docker": {
        "label": "Technology",
        "aliases": ["Docker"],
        "category": "Containerization",
    },
    "Kubernetes": {
        "label": "Technology",
        "aliases": ["Kubernetes", "GKE", "EKS", "AKS"],
        "category": "Container orchestration",
    },
    "Vault": {
        "label": "Technology",
        "aliases": ["Vault"],
        "category": "Secrets management",
    },
    # Architecture components and services
    "API Gateway": {
        "label": "Component",
        "aliases": ["API gateway", "gateway"],
        "category": "API entry point",
    },
    "Auth & Identity Service": {
        "label": "Service",
        "aliases": ["Auth & Identity Service", "Identity Service"],
        "category": "Authentication and identity",
    },
    "User Profile Service": {
        "label": "Service",
        "aliases": ["User Profile Service"],
        "category": "User profiles",
    },
    "Catalog Service": {
        "label": "Service",
        "aliases": ["Catalog Service"],
        "category": "Music metadata",
    },
    "Search & Discovery Index Service": {
        "label": "Service",
        "aliases": ["Search & Discovery Index Service"],
        "category": "Search and discovery",
    },
    "Library & Playlist Service": {
        "label": "Service",
        "aliases": ["Library & Playlist Service"],
        "category": "Libraries and playlists",
    },
    "Playback & Session Service": {
        "label": "Service",
        "aliases": ["Playback & Session Service"],
        "category": "Playback and sessions",
    },
    "Recommendation Service": {
        "label": "Service",
        "aliases": ["Recommendation Service"],
        "category": "Personalization and recommendations",
    },
    "Notification Service": {
        "label": "Service",
        "aliases": ["Notification Service"],
        "category": "Notifications",
    },
    "Media Processor Service": {
        "label": "Service",
        "aliases": ["Media Processor Service"],
        "category": "Media processing",
    },
    # Security
    "JWT": {
        "label": "Protocol",
        "aliases": ["JWT"],
        "category": "Token format",
    },
    "RBAC": {
        "label": "Feature",
        "aliases": ["RBAC", "role-based access control"],
        "category": "Access control",
    },
    "HTTPS": {
        "label": "Protocol",
        "aliases": ["HTTPS"],
        "category": "Secure transport",
    },
    "Rate Limiting": {
        "label": "Feature",
        "aliases": ["rate limiting", "rate limits"],
        "category": "Request protection",
    },
    "Audit Logging": {
        "label": "Feature",
        "aliases": ["audit logging", "audit logs"],
        "category": "Security auditing",
    },

}


def _contains_alias(text: str, alias: str) -> bool:
    """Match a complete term rather than part of a longer word."""
    pattern = rf"(?<!\w){re.escape(alias)}(?!\w)"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def extract_entities(chunks: list[dict]) -> list[dict]:
    """
    Extract catalogued entities from text chunks.

    Each result retains source, page, and chunk ID for traceability.
    Only explicit catalog matches are emitted; no relationships are inferred.
    """
    entities = []
    seen = set()

    for chunk in chunks:
        text = chunk.get("text", "")

        if not isinstance(text, str) or not text.strip():
            continue

        for canonical_name, definition in sorted(
            ENTITY_CATALOG.items(),
            key=lambda item: max(map(len, item[1]["aliases"])),
            reverse=True,
        ):
            matched_aliases = [
                alias
                for alias in definition["aliases"]
                if _contains_alias(text, alias)
            ]

            if not matched_aliases:
                continue

            identity = (
                canonical_name,
                chunk.get("source"),
                chunk.get("page"),
                chunk.get("chunk_id"),
            )

            if identity in seen:
                continue

            seen.add(identity)

            entities.append({
                "name": canonical_name,
                "label": definition["label"],
                "category": definition["category"],
                "source": chunk.get("source"),
                "page": chunk.get("page"),
                "chunk_id": chunk.get("chunk_id"),
                "matched_aliases": matched_aliases,
            })

    return entities
