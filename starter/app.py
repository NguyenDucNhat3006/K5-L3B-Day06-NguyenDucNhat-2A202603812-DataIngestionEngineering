"""Streamlit UI for live pgvector retrieval and saved experiment artifacts."""

import json
import os
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from src.embeddings import MyEmbedder  # noqa: E402
from src.run import DEFAULT_MODEL, load_questions  # noqa: E402
from src.store import PgVectorStore  # noqa: E402

RESULTS = HERE / "results"
BENCHMARK_PATH = HERE / "benchmark" / "questions.json"
MANIFEST_PATH = HERE / "data" / "manifest.json"
load_dotenv(HERE / ".env")

st.set_page_config(page_title="Calculus Retrieval Lab", page_icon="∑", layout="wide")


@st.cache_resource(show_spinner="Loading multilingual embedding model…")
def get_embedder(model_name: str) -> MyEmbedder:
    return MyEmbedder(model_name)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def get_store() -> PgVectorStore:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is missing. Copy .env.example to .env and configure PostgreSQL.")
    model_name = os.getenv("EMBEDDING_MODEL") or DEFAULT_MODEL
    return PgVectorStore(url, get_embedder(model_name).dimensions)


def top_chunks(question: str, strategy: str, top_k: int, document_id: str | None) -> dict[str, list[dict[str, Any]]]:
    model_name = os.getenv("EMBEDDING_MODEL") or DEFAULT_MODEL
    vector = get_embedder(model_name).embed_query(question)
    store = get_store()
    strategies = {
        "A": ("A_fixed", "A_fixed", None),
        "B": ("B_paragraph", "B_paragraph", None),
        "C": ("C_paragraph_filter", "B_paragraph", {"document_id": document_id} if document_id else None),
    }
    selected = list(strategies) if strategy == "Compare all" else [strategy]
    return {
        strategies[name][0]: store.search(strategies[name][1], vector, top_k=top_k, filters=strategies[name][2])
        for name in selected
    }


def show_chunk(rank: int, hit: dict[str, Any]) -> None:
    meta = hit.get("metadata", {})
    st.markdown(f"**#{rank} · cosine similarity {hit['score']:.4f}**")
    cols = st.columns(4)
    for col, label, value in zip(cols, ("Source", "Page", "Document ID", "Chunk ID"), (meta.get("source", "—"), meta.get("page", "—"), meta.get("document_id", "—"), hit.get("chunk_id", "—"))):
        col.caption(label)
        col.write(value)
    with st.container(border=True):
        st.text(hit.get("text", ""))


def page_playground() -> None:
    st.title("Retrieval playground", anchor=False)
    st.caption("Search the indexed calculus corpus and inspect the evidence returned by pgvector.")
    st.info("This page performs retrieval only. It does not generate an LLM answer.")
    try:
        manifest = read_json(MANIFEST_PATH)
        options = [Path(item["source"]).stem for item in manifest]
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        options = []
    with st.form("retrieval"):
        question = st.text_input("Câu hỏi bằng tiếng Việt", placeholder="Ví dụ: Điều kiện để hàm khả vi tại điểm là gì?")
        c1, c2, c3 = st.columns([2, 1, 2])
        strategy = c1.selectbox("Chiến lược", ["A", "B", "C", "Compare all"])
        top_k = c2.number_input("Top-K", min_value=1, max_value=20, value=3, step=1)
        selected_doc = c3.selectbox("Document filter (C)", ["Không lọc", *options])
        submitted = st.form_submit_button("Retrieve evidence", type="primary", width="stretch")
    if submitted:
        if not question.strip():
            st.warning("Nhập câu hỏi trước khi truy xuất.")
            return
        if strategy == "C" and selected_doc == "Không lọc":
            st.caption("C đang chạy không có document filter.")
        try:
            results = top_chunks(question, strategy, int(top_k), selected_doc if selected_doc != "Không lọc" else None)
            if not any(results.values()):
                st.warning("Không có kết quả. Kiểm tra database và chạy lệnh index.")
            for label, hits in results.items():
                st.subheader(label)
                if not hits:
                    st.caption("Không tìm thấy chunk phù hợp.")
                for rank, hit in enumerate(hits, 1):
                    with st.expander(f"#{rank} · {hit['score']:.4f} · {hit.get('metadata', {}).get('source')} · p.{hit.get('metadata', {}).get('page')}", expanded=(rank == 1)):
                        show_chunk(rank, hit)
        except Exception as exc:
            st.error(f"Retrieval failed: {exc}")


def page_dashboard() -> None:
    st.title("Experiment dashboard", anchor=False)
    st.caption("Read-only view of saved retrieval metrics and manual audits.")
    try:
        metrics = read_json(RESULTS / "metrics.json")
        coherence = pd.read_csv(RESULTS / "chunk_coherence_audit.csv")
        grounding = pd.read_csv(RESULTS / "grounding_audit.csv")
        experiments = pd.read_csv(HERE / "artifacts" / "experiment_log.csv")
        questions = read_json(BENCHMARK_PATH)
        c_traces = read_json(RESULTS / "traces_C_paragraph_filter.json")
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.error(f"Saved evaluation artifacts could not be loaded: {exc}")
        return
    names = {"A_fixed": "A · Fixed size", "B_paragraph": "B · Paragraph", "C_paragraph_filter": "C · Filtered"}
    summary_rows = []
    for key, data in metrics.items():
        for split in ("overall", "dev", "test"):
            item = data[split]
            summary_rows.append({"Strategy": names.get(key, key), "Split": split.title(), "Hit@1": item["hit_at_1"], "Gold-location Hit@3": item["evidence_hit_at_3"], "N": item["questions"]})
    summary = pd.DataFrame(summary_rows)
    st.subheader("Retrieval metrics")
    split = st.segmented_control("Split", ["Overall", "Dev", "Test"], default="Overall") or "Overall"
    selected = summary[summary.Split == split]
    chart = selected.set_index("Strategy")[["Hit@1", "Gold-location Hit@3"]]
    st.bar_chart(chart, y_label="Share of questions", alt=f"Hit at one and gold-location hit at three for the {split.lower()} split")
    st.dataframe(selected, hide_index=True, alt="Retrieval metrics by strategy", column_config={"Hit@1": st.column_config.NumberColumn(format="%.1%"), "Gold-location Hit@3": st.column_config.NumberColumn(format="%.1%")})
    st.caption("Gold-location Hit@3 counts any top-3 chunk from the accepted source and page. Manual grounding checks whether the retrieved evidence contains enough context to answer; these are different measurements.")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Chunk coherence")
        if {"A_score", "B_score"}.issubset(coherence.columns):
            counts = pd.DataFrame({"A": [coherence.A_score.sum()], "B / C": [coherence.B_score.sum()]}, index=[f"Points / {len(coherence) * 2}"])
            st.bar_chart(counts, alt="Manual chunk coherence points by strategy")
            st.caption(f"Manual scores across {len(coherence)} sampled source locations. B and C share paragraph chunking.")
        st.dataframe(coherence, hide_index=True, alt="Chunk coherence audit labels and reasons")
    with c2:
        st.subheader("Manual grounding · test")
        grouped = grounding.groupby("strategy").grounding_score.agg([("Mean / 2", "mean"), ("Audited questions", "count")]).rename(index=names)
        st.bar_chart(grouped[["Mean / 2"]], alt="Manual grounding score by strategy")
        earned = int(grounding.grounding_score.sum())
        possible = int(grounding.max_score.sum())
        st.caption(f"Manual audit total: {earned}/{possible} points across {len(grounding)} test questions per strategy.")
        st.dataframe(grounding, hide_index=True, alt="Manual grounding audit by question and strategy")
    st.subheader("Metadata filter compliance")
    filtered_questions = {item["id"] for item in questions if item.get("filter")}
    compliance_total = compliance_pass = 0
    for trace in c_traces:
        if trace["id"] not in filtered_questions:
            continue
        compliance_total += 1
        requested = questions[[item["id"] for item in questions].index(trace["id"])]["filter"]
        hits = trace.get("hits", [])
        if hits and all(all(hit.get(key) == value for key, value in requested.items()) for hit in hits):
            compliance_pass += 1
    st.metric("C · Filtered top-3 compliance", f"{compliance_pass} / {compliance_total}", help="Calculated from saved C traces and the benchmark's question-provided filters.")
    st.subheader("Experiment settings and measurements")
    st.dataframe(experiments, hide_index=True, alt="Experiment settings and measured results")


def page_explorer() -> None:
    st.title("Benchmark explorer", anchor=False)
    st.caption("Browse the fixed benchmark and saved top-3 retrieval traces.")
    questions = load_questions()
    split = st.selectbox("Split", ["All", "dev", "test"])
    choices = [q for q in questions if split == "All" or q["split"] == split]
    label_to_id = {f"{q['id']} · {q['question']}": q["id"] for q in choices}
    chosen = st.selectbox("Question", list(label_to_id))
    qid = label_to_id[chosen]
    question = next(item for item in questions if item["id"] == qid)
    left, right = st.columns(2)
    with left:
        st.markdown(f"**Split:** `{question['split']}` · **Type:** `{question.get('type', '—')}`")
        st.markdown("**Expected answer**")
        st.write(question["expected_answer"])
        st.markdown("**Gold evidence**")
        st.write(question["evidence_quote"])
    with right:
        st.markdown("**Gold location**")
        st.write(f"{question['gold_source_id']} · page {question['gold_page']}")
        st.markdown("**Metadata filter**")
        st.json(question.get("filter"))
    st.divider()
    tabs = st.tabs(["A · Fixed", "B · Paragraph", "C · Filtered"])
    trace_paths = {"A · Fixed": "traces_A_fixed.json", "B · Paragraph": "traces_B_paragraph.json", "C · Filtered": "traces_C_paragraph_filter.json"}
    for tab, (strategy_name, file_name) in zip(tabs, trace_paths.items()):
        with tab:
            try:
                trace = next(t for t in read_json(RESULTS / file_name) if t["id"] == qid)
            except (OSError, StopIteration, KeyError, json.JSONDecodeError) as exc:
                st.error(f"Saved trace missing or invalid: {exc}")
                continue
            st.write(f"**Hit@1:** {trace['hit_at_1']} · **Gold-location Hit@3:** {trace['evidence_hit_at_3']} · **first gold rank:** {trace['first_gold_rank'] or 'not in top 3'}")
            for hit in trace.get("hits", []):
                with st.expander(f"#{hit['rank']} · {hit['score']:.4f} · {hit['source']} · p.{hit['page']}" + (" · gold location" if hit.get("matches_gold_location") else ""), expanded=(hit["rank"] == 1)):
                    show_chunk(hit["rank"], {"chunk_id": hit["chunk_id"], "score": hit["score"], "text": hit["text"], "metadata": hit})
            if not trace.get("hits"):
                st.caption("Saved trace contains no retrieved chunks.")


st.sidebar.title("∑ Calculus Retrieval Lab")
page = st.sidebar.radio("Navigate", ["Retrieval Playground", "Experiment Dashboard", "Benchmark Explorer"])
st.sidebar.markdown("---")
st.sidebar.caption("5 calculus PDFs · 249 content pages · 20 locked questions")
if page == "Retrieval Playground":
    page_playground()
elif page == "Experiment Dashboard":
    page_dashboard()
else:
    page_explorer()
