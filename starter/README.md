# Day 06 · Data Ingestion Engineering

Python 3.12 pipeline for five calculus PDFs, local multilingual embeddings, PostgreSQL 16 + pgvector, and a Streamlit retrieval demo. The saved benchmark is a locked 20-question set (15 dev, 5 test). The app retrieves evidence; it does not generate answers.

## Setup

From this directory, create a Python 3.12 environment and install dependencies:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
docker compose -p day6_student_starter up -d --wait
python scripts/check_setup.py
```

The configured local database uses `127.0.0.1:55433`. Do not commit `.env`. The embedding model downloads on first use and runs locally; it is 384-dimensional. Docker must be running for database retrieval.

## Indexing and evaluation

Run commands from `starter/`:

```powershell
# Read PDFs, chunk, embed, and upsert A and B indexes. C reuses B.
python -m src.run index

# Query the existing indexes and replace saved benchmark metrics/traces.
# This does not embed corpus chunks.
python -m src.run evaluate

# Explicitly run both steps (the default is retained for compatibility).
python -m src.run all
```

Indexing uses fixed windows (A: 500 characters, overlap 100) and paragraph-aware chunks (B: max 500 characters). Stable chunk IDs and PostgreSQL upserts make reruns idempotent; obsolete chunks for manifest documents are removed. C performs question-provided document filters in the SQL query over the B index. The benchmark JSON is read-only during evaluation. Evaluation overwrites generated `results/metrics.json` and `results/traces_*.json`; preserve/copy these before deliberately creating a new run if you need the earlier outputs.

## Streamlit demo

```powershell
streamlit run app.py
```

Pages: Retrieval Playground (live pgvector search, optional C document filter), Experiment Dashboard (saved metrics and audits), and Benchmark Explorer (read-only questions and saved A/B/C traces). The local embedding model is cached for the app session. Retrieval and LLM answer generation are separate; this project does not call an answer-generation model.

## Reproducibility and artifacts

- Corpus manifest: `data/manifest.json`; PDFs are in `data/raw/` and are intentionally not tracked.
- Locked benchmark: `benchmark/questions.json`.
- Saved results and human audits: `results/`.
- Experiment configuration and measurements: `artifacts/experiment_log.csv`.
- One can verify Python imports without database access with `python -m unittest discover -s tests -v`.
- PostgreSQL pgvector uses cosine distance (`<=>`) and exact ordering for this small corpus; filtering is applied before ranking in SQL.

Physical page numbers are preserved. PDF text extraction can damage mathematical notation, matrices, and multi-line formulas; see the individual report and the audited failures. Source redistribution permissions are not established in the manifest and must be verified before sharing PDFs or extracted content; evaluation regenerates the traces locally.

- Original PDF files are kept locally and are not committed.
- Retrieval traces and grounding audit artifacts are included
  in the repository for academic evaluation and reproducibility.
- Use and redistribution are subject to the permissions
  documented in the corpus manifest.