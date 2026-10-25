
def chunk_pages(
    pages: list[dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[dict]:
    """
    Split cleaned PDF pages into overlapping text chunks.

    Each chunk preserves its source filename and page number.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative and smaller than chunk_size."
        )

    chunks = []
    step = chunk_size - chunk_overlap

    for page_data in pages:
        text = page_data.get("text", "").strip()

        if not text:
            continue

        for start in range(0, len(text), step):
            chunk_text = text[start : start + chunk_size].strip()

            if not chunk_text:
                continue

            chunks.append({
                "source": page_data["source"],
                "page": page_data["page"],
                "chunk_id": len(chunks),
                "text": chunk_text,
            })

            if start + chunk_size >= len(text):
                break

    return chunks
