"""Index the corpus and/or run the locked A/B/C retrieval evaluation.

Run from starter/: ``python -m src.run index`` or ``python -m src.run evaluate``.
Evaluation never re-embeds corpus chunks; C searches the B index with SQL filters.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .chunking import by_paragraph, fixed_size
from .embeddings import MyEmbedder
from .evaluation import evaluate_query, summarize
from .ingestion import load_pdf
from .store import PgVectorStore

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR, RAW_DIR = ROOT / "data", ROOT / "data" / "raw"
MANIFEST_PATH = DATA_DIR / "manifest.json"
BENCHMARK_PATH = ROOT / "benchmark" / "questions.json"
RESULTS_DIR = ROOT / "results"
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8-sig") as stream:
            return json.load(stream)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def load_manifest() -> list[dict[str, Any]]:
    manifest = read_json(MANIFEST_PATH)
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("Manifest must be a non-empty JSON array")
    seen: set[str] = set()
    for index, item in enumerate(manifest):
        if not isinstance(item, dict) or not item.get("source") or not item.get("title"):
            raise ValueError(f"Manifest entry {index} needs source and title")
        source = Path(item["source"])
        if source.name != str(source) or source.suffix.lower() != ".pdf":
            raise ValueError(f"Manifest source must be a PDF filename: {item['source']!r}")
        if item.get("source_id", source.stem) != source.stem:
            raise ValueError(f"source_id must match the PDF stem for {item['source']!r}")
        if source.stem in seen:
            raise ValueError(f"Duplicate document ID: {source.stem}")
        seen.add(source.stem)
        if not (RAW_DIR / source).is_file():
            raise FileNotFoundError(f"PDF not found: {RAW_DIR / source}")
    return manifest


def load_questions() -> list[dict[str, Any]]:
    questions = read_json(BENCHMARK_PATH)
    if not isinstance(questions, list) or len(questions) < 20:
        raise ValueError("Benchmark must contain at least 20 questions")
    ids: set[str] = set()
    splits = {"dev": 0, "test": 0}
    for index, question in enumerate(questions):
        required = ("id", "split", "question", "gold_source_id", "gold_page", "evidence_quote", "expected_answer")
        if not isinstance(question, dict) or any(not question.get(key) for key in required):
            raise ValueError(f"Benchmark question {index} is missing required fields")
        if question["id"] in ids:
            raise ValueError(f"Duplicate benchmark question ID: {question['id']}")
        ids.add(question["id"])
        if question["split"] not in splits:
            raise ValueError(f"Unsupported split in {question['id']}: {question['split']}")
        splits[question["split"]] += 1
        filters = question.get("filter")
        if filters is not None and (not isinstance(filters, dict) or not filters):
            raise ValueError(f"Filter for {question['id']} must be null or a non-empty object")
        if filters is not None:
            allowed = {"document_id", "source", "page"}
            if set(filters) - allowed or any(value is None or value == "" for value in filters.values()):
                raise ValueError(f"Filter for {question['id']} contains unsupported keys or empty values")
    if splits["dev"] < 15 or splits["test"] < 5:
        raise ValueError(f"Benchmark needs >=15 dev and >=5 test questions; got {splits}")
    return questions


def make_store(model_name: str) -> tuple[MyEmbedder, PgVectorStore]:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL is missing; configure starter/.env")
    embedder = MyEmbedder(model_name)
    return embedder, PgVectorStore(database_url, embedder.dimensions)


def load_pages(manifest: list[dict[str, Any]]):
    pages = []
    document_ids = []
    for item in manifest:
        source = item["source"]
        document_id = Path(source).stem
        document_ids.append(document_id)
        doc_pages = load_pdf(RAW_DIR / source, document_id, source)
        for page in doc_pages:
            page.metadata.update({key: item.get(key) for key in ("chapter", "title", "author", "institution")})
        pages.extend(doc_pages)
    content_pages = [page for page in pages if page.text.strip()]
    if len(content_pages) < 30:
        raise ValueError(f"Corpus has {len(content_pages)} content pages; at least 30 are required")
    return content_pages, document_ids


def index_corpus(embedder: MyEmbedder, store: PgVectorStore) -> None:
    manifest = load_manifest()
    pages, document_ids = load_pages(manifest)
    chunks_a = [chunk for page in pages for chunk in fixed_size(page, size=500, overlap=100)]
    chunks_b = [chunk for page in pages for chunk in by_paragraph(page, max_size=500)]
    if not chunks_a or not chunks_b:
        raise ValueError("Chunking produced no chunks; inspect PDF text extraction")
    store.create_schema()
    for strategy, chunks in (("A_fixed", chunks_a), ("B_paragraph", chunks_b)):
        vectors = embedder.embed_documents([chunk.text for chunk in chunks])
        if len(vectors) != len(chunks):
            raise RuntimeError(f"Embedding provider returned {len(vectors)} vectors for {len(chunks)} chunks")
        store.upsert(strategy, chunks, vectors, document_ids=document_ids)
        print(f"Indexed {strategy}: {len(chunks)} chunks")
    print("C uses the B_paragraph vectors and applies metadata filters at query time.")


def run_benchmark(questions: list[dict[str, Any]], embedder: MyEmbedder, store: PgVectorStore) -> dict[str, list[dict[str, Any]]]:
    traces = {name: [] for name in ("A_fixed", "B_paragraph", "C_paragraph_filter")}
    for index, question in enumerate(questions, start=1):
        print(f"[{index}/{len(questions)}] {question['id']}: {question['question']}")
        vector = embedder.embed_query(question["question"])
        for strategy, index_name, filters in (
            ("A_fixed", "A_fixed", None),
            ("B_paragraph", "B_paragraph", None),
            ("C_paragraph_filter", "B_paragraph", question.get("filter")),
        ):
            hits = store.search(index_name, vector, top_k=3, filters=filters)
            trace = evaluate_query(question, hits)
            trace["strategy"] = strategy
            traces[strategy].append(trace)
    return traces


def save_evaluation(traces: dict[str, list[dict[str, Any]]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for strategy, strategy_traces in traces.items():
        (RESULTS_DIR / f"traces_{strategy}.json").write_text(
            json.dumps(strategy_traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        metrics[strategy] = summarize(strategy_traces)
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("index", "evaluate", "all"), nargs="?", default="all")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    model_name = os.getenv("EMBEDDING_MODEL") or DEFAULT_MODEL
    embedder, store = make_store(model_name)
    if args.command in ("index", "all"):
        index_corpus(embedder, store)
    if args.command in ("evaluate", "all"):
        store.create_schema()
        counts = {name: store.count_chunks(name) for name in ("A_fixed", "B_paragraph")}
        if any(count == 0 for count in counts.values()):
            raise RuntimeError(
                f"Indexes are missing or empty ({counts}); run `python -m src.run index` first. "
                "Saved benchmark outputs were not changed."
            )
        traces = run_benchmark(load_questions(), embedder, store)
        save_evaluation(traces)
        print(json.dumps({key: value["overall"] for key, value in ((k, summarize(v)) for k, v in traces.items())}, indent=2))
        print(f"Evaluation artifacts saved in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
