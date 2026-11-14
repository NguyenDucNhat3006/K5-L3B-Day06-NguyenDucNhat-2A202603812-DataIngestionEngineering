import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import csv

from src.chunking import by_paragraph, fixed_size
from src.evaluation import evaluate_query, summarize
from src.ingestion import normalize_text
from src.models import DocumentPage
from src.run import load_questions, read_json
from src.store import PgVectorStore


class FakeCursor:
    def __init__(self):
        self.sql = ""
        self.params = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, sql, params):
        self.sql, self.params = sql, params

    def fetchall(self):
        return []


class FakeConnection:
    def __init__(self, cursor):
        self.fake_cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def cursor(self):
        return self.fake_cursor


class PipelineTests(unittest.TestCase):
    def test_utf8_bom_json_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bom.json"
            path.write_bytes(b"\xef\xbb\xbf[{\"ok\": true}]")
            self.assertEqual(read_json(path), [{"ok": True}])

    def test_normalization_preserves_paragraphs_and_unicode(self):
        self.assertEqual(normalize_text(" A  \r\n\r\n B  "), "A\n\nB")

    def test_chunkers_validate_parameters_and_keep_traceability(self):
        page = DocumentPage("doc", "doc.pdf", 4, "alpha beta gamma delta", {"chapter": 1})
        fixed = fixed_size(page, size=12, overlap=2)
        paragraphs = by_paragraph(page, max_size=12)
        self.assertTrue(fixed)
        self.assertTrue(paragraphs)
        self.assertEqual(fixed[0].document_id, "doc")
        self.assertEqual(paragraphs[0].page, 4)
        with self.assertRaises(ValueError):
            fixed_size(page, size=4, overlap=4)

    def test_gold_location_metrics_and_trace(self):
        question = {"id": "q", "split": "dev", "question": "x", "gold_source_id": "doc", "gold_page": 3}
        hit = {"chunk_id": "c", "text": "evidence", "score": 0.9, "metadata": {"document_id": "doc", "page": 3, "source": "doc.pdf"}}
        trace = evaluate_query(question, [hit])
        self.assertTrue(trace["hit_at_1"])
        self.assertTrue(trace["evidence_hit_at_3"])
        self.assertEqual(summarize([trace])["overall"]["hit_at_1_count"], 1)

    def test_metadata_filter_is_parameterized_in_sql_before_ordering(self):
        cursor = FakeCursor()
        store = PgVectorStore("postgresql://unused", 2)
        with patch.object(store, "_connect", return_value=FakeConnection(cursor)):
            store.search("B_paragraph", [0.1, 0.2], filters={"document_id": "ch1"})
        self.assertIn("document_id = %s", cursor.sql)
        self.assertLess(cursor.sql.index("WHERE"), cursor.sql.index("ORDER BY"))
        self.assertEqual(cursor.params[2], "ch1")

    def test_locked_benchmark_has_20_questions_and_split(self):
        questions = load_questions()
        self.assertEqual(len(questions), 20)
        self.assertEqual(sum(q["split"] == "dev" for q in questions), 15)
        self.assertEqual(sum(q["split"] == "test" for q in questions), 5)

    def test_saved_metrics_match_preserved_traces_and_audits(self):
        root = Path(__file__).resolve().parents[1]
        metrics = read_json(root / "results" / "metrics.json")
        for key in ("A_fixed", "B_paragraph", "C_paragraph_filter"):
            traces = read_json(root / "results" / f"traces_{key}.json")
            self.assertEqual(len(traces), metrics[key]["overall"]["questions"])
            for split in ("overall", "dev", "test"):
                subset = traces if split == "overall" else [t for t in traces if t["split"] == split]
                self.assertEqual(sum(bool(t["hit_at_1"]) for t in subset), metrics[key][split]["hit_at_1_count"])
                self.assertEqual(sum(bool(t["evidence_hit_at_3"]) for t in subset), metrics[key][split]["evidence_hit_at_3_count"])
        with (root / "results" / "grounding_audit.csv").open(encoding="utf-8-sig", newline="") as source:
            rows = list(csv.DictReader(source))
        self.assertEqual(len(rows), 15)


if __name__ == "__main__":
    unittest.main()
