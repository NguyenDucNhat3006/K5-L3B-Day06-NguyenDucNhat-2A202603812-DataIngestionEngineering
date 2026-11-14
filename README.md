# Day 06 · Data Ingestion Engineering

Retrieval engineering lab using five calculus PDFs (249 content pages), three controlled configurations (A fixed-size, B paragraph-aware, C B plus question-specific SQL metadata filters), multilingual sentence-transformer embeddings, and PostgreSQL 16 + pgvector.

## Quick start

See [`starter/README.md`](starter/README.md) for Python 3.12, Docker, database, and model setup. From `starter/`:

```powershell
python -m src.run index       # index corpus; C reuses B vectors
python -m src.run evaluate    # benchmark existing indexes; no corpus re-embedding
streamlit run app.py          # live retrieval playground and artifact dashboards
```

To run both pipeline stages explicitly: `python -m src.run all`. Evaluation writes metrics and A/B/C top-3 traces under `starter/results/`. The benchmark in `starter/benchmark/questions.json` is locked and read-only in the UI. Do not tune on the five test questions.

## Verified saved results

The saved metrics record 20 questions (15 dev, 5 test); A, B, and C each have overall Hit@1 **6/20 (0.30)** and gold-location Hit@3 **10/20 (0.50)**. Test grounding audits are **6/10** for each strategy, and C filter compliance is **4/4**. The coherence audit scores A **17/20** and B/C **16/20**. These measurements describe this corpus and small benchmark only. The dashboard reads values from saved files rather than embedding scores in its code.

PDF extraction has visible formula/layout errors, including an incomplete differentiability definition (q14), a Riemann-sum definition missing from top-3 evidence (q20), and damaged matrix/formula layout recorded in the coherence audit. These affect evidence completeness. No LLM answer-generation evaluation was performed.

## Submission and data notes

- Pipeline and demo: [`starter/`](starter/README.md)
- Student record: [`STUDENT.md`](STUDENT.md)
- Experiment log: [`artifacts/experiment_log.csv`](artifacts/experiment_log.csv)
- Individual report: [`report/individual/2A202603812_NguyenDucNhat.md`](report/individual/2A202603812_NguyenDucNhat.md)
- Corpus permissions are unknown and marked for verification. PDFs are not committed; verify redistribution rights before sharing the source PDFs or extracted traces.
- `.env`, credentials, and local databases must remain untracked. No GitHub push or deployment is part of this repository workflow.

## Lab references

The original lab brief and evaluation instructions remain available: [GUIDE](GUIDE.md), [locked benchmark specification](BENCHMARK.md), [rubric](RUBRIC.md), [data and API rules](RULES.md), [submission checklist](SUBMISSION.md), and [four-hour checkpoints](CHECKPOINTS.md). The pipeline architecture illustration is in [`assets/`](assets/).
