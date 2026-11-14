"""Evaluate retrieved evidence and preserve per-query traces."""

from typing import Any


def _gold_locations(question: dict[str, Any]) -> list[tuple[str, int]]:
    """Return all acceptable (source_id, page) gold locations."""
    alternatives = question.get("gold_alternatives")

    if alternatives:
        return [
            (item["gold_source_id"], int(item["gold_page"]))
            for item in alternatives
        ]

    return [
        (
            question["gold_source_id"],
            int(question["gold_page"]),
        )
    ]


def _matches_gold(
    hit: dict[str, Any],
    gold_locations: list[tuple[str, int]],
) -> bool:
    """Check whether a retrieved chunk comes from an accepted gold location."""
    metadata = hit.get("metadata", {})

    document_id = metadata.get("document_id")
    page = metadata.get("page")

    return any(
        document_id == gold_source_id and page == gold_page
        for gold_source_id, gold_page in gold_locations
    )


def evaluate_query(
    question: dict[str, Any],
    hits: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate Hit@1 and Evidence Hit@3 while preserving retrieval trace.

    Automatic retrieval correctness is based on traceable gold source/page.
    Human grounding and chunk-coherence judgments are intentionally not inferred.
    """
    gold_locations = _gold_locations(question)

    top3 = hits[:3]

    hit1 = bool(top3) and _matches_gold(top3[0], gold_locations)
    evidence_hit3 = any(
        _matches_gold(hit, gold_locations)
        for hit in top3
    )

    first_gold_rank = None

    for rank, hit in enumerate(top3, start=1):
        if _matches_gold(hit, gold_locations):
            first_gold_rank = rank
            break

    trace_hits = []

    for rank, hit in enumerate(top3, start=1):
        metadata = hit.get("metadata", {})

        trace_hits.append(
            {
                "rank": rank,
                "chunk_id": hit.get("chunk_id"),
                "score": hit.get("score"),
                "document_id": metadata.get("document_id"),
                "source": metadata.get("source"),
                "page": metadata.get("page"),
                "text": hit.get("text"),
                "matches_gold_location": _matches_gold(
                    hit,
                    gold_locations,
                ),
            }
        )

    return {
        "id": question["id"],
        "split": question["split"],
        "question": question["question"],
        "type": question.get("type"),
        "filter": question.get("filter"),
        "gold_source_id": question["gold_source_id"],
        "gold_page": question["gold_page"],
        "expected_answer": question.get("expected_answer"),
        "evidence_quote": question.get("evidence_quote"),
        "hit_at_1": hit1,
        "evidence_hit_at_3": evidence_hit3,
        "first_gold_rank": first_gold_rank,
        "hits": trace_hits,
    }


def summarize(
    traces: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate Hit@1 and Evidence Hit@3 overall and by split."""
    def metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(items)

        if total == 0:
            return {
                "questions": 0,
                "hit_at_1_count": 0,
                "hit_at_1": 0.0,
                "evidence_hit_at_3_count": 0,
                "evidence_hit_at_3": 0.0,
            }

        hit1_count = sum(
            1 for item in items
            if item["hit_at_1"]
        )

        hit3_count = sum(
            1 for item in items
            if item["evidence_hit_at_3"]
        )

        return {
            "questions": total,
            "hit_at_1_count": hit1_count,
            "hit_at_1": hit1_count / total,
            "evidence_hit_at_3_count": hit3_count,
            "evidence_hit_at_3": hit3_count / total,
        }

    dev = [
        trace for trace in traces
        if trace.get("split") == "dev"
    ]

    test = [
        trace for trace in traces
        if trace.get("split") == "test"
    ]

    return {
        "overall": metrics(traces),
        "dev": metrics(dev),
        "test": metrics(test),
    }