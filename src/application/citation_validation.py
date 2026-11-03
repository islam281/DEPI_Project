"""Fast structural reference checks; NO semantic LLM judge or human review."""

def validate_structural_citations(response: dict) -> dict:
    """Validate source references without claiming semantic verification.

    Rules:
    - References must be unique in the supplied source list.
    - Each cited reference must exist in the source list.
    - Cited sources must have text and page metadata.
    - Answered responses must contain at least one citation.
    - Abstentions should not contain citations.

    This function performs no retrieval, inference, or LLM call.
    """
    sources = response.get("sources") or []
    citations = response.get("citations") or []
    status = response.get("status")

    errors = []
    source_lookup = {}

    for source in sources:
        reference = str(source.get("reference", "")).strip()
        if not reference:
            errors.append("A source is missing its reference ID")
            continue
        if reference in source_lookup:
            errors.append(f"Duplicate source reference: {reference}")
        source_lookup[reference] = source

    if status == "answered" and not citations:
        errors.append("Answered response has no citations")
    if status == "insufficient_evidence" and citations:
        errors.append("Abstention must not contain citations")
    if len(citations) != len(set(citations)):
        errors.append("Repeated citation IDs in the answer")

    for reference in citations:
        source = source_lookup.get(reference)
        if source is None:
            errors.append(f"Unknown source reference: {reference}")
            continue

        if not str(source.get("content") or "").strip():
            errors.append(f"Empty source text for {reference}")

        metadata = source.get("metadata") or {}
        page = source.get("page_number", metadata.get("page_number", metadata.get("page")))
        if page in (None, ""):
            errors.append(f"Missing page metadata for {reference}")
        else:
            try:
                if int(page) < 1:
                    errors.append(f"Invalid page number for {reference}: {page}")
            except (TypeError, ValueError):
                errors.append(f"Invalid page number for {reference}: {page}")

    return {
        "valid": not errors,
        "decision": "RETURN_WITHOUT_VERIFIED_BADGE" if not errors else "WITHHOLD_ANSWER",
        "source_integrity": "VALID" if not errors else "INVALID",
        "semantic_status": "NOT_VERIFIED",
        "validated_refs": len(citations) if not errors else sum(
            1 for reference in citations if reference in source_lookup
        ),
        "errors": errors,
    }
