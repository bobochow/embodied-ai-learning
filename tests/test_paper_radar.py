import datetime as dt
import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "papers" / "update.py"
SPEC = importlib.util.spec_from_file_location("paper_radar", MODULE_PATH)
paper_radar = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = paper_radar
assert SPEC.loader is not None
SPEC.loader.exec_module(paper_radar)


class PaperRadarTests(unittest.TestCase):
    def paper(self, **overrides):
        values = {
            "arxiv_id": "2609.00001",
            "title": "A Vision-Language-Action Model for Robot Manipulation",
            "abstract": "We train a robot policy and evaluate it on a real robot.",
            "authors": ["A. Researcher"],
            "categories": ["cs.RO"],
            "published": "2026-09-09T00:00:00Z",
            "updated": "2026-09-09T00:00:00Z",
            "sources": ["daily_arxiv"],
            "source_sections": ["Vision Language Action"],
        }
        values.update(overrides)
        return paper_radar.Paper(**values)

    def test_canonical_arxiv_id_removes_version(self):
        self.assertEqual(
            paper_radar.canonical_arxiv_id("https://arxiv.org/abs/2609.09158v2"),
            "2609.09158",
        )

    def test_extracts_only_requested_sections(self):
        markdown = """
## Vision Language Action
- [A](https://arxiv.org/abs/2609.00001v1)
## world model
- [B](https://arxiv.org/abs/2609.00002)
## world action model
- [C](https://arxiv.org/abs/2609.00003)
"""
        result = paper_radar.extract_ids_by_section(
            markdown,
            ["Vision Language Action", "world action model"],
            10,
        )
        self.assertEqual(result["Vision Language Action"], ["2609.00001"])
        self.assertEqual(result["world action model"], ["2609.00003"])

    def test_scores_embodied_paper_above_threshold(self):
        paper = paper_radar.classify_and_score(self.paper(), [])
        self.assertGreaterEqual(paper.relevance_score, 5)
        self.assertEqual(paper.track, "vla")

    def test_explicit_vla_title_is_not_demoted_by_benchmark_abstract(self):
        paper = self.paper(
            title="ZETA: Cross-Embodiment VLA Transfer",
            abstract="We introduce a benchmark and dataset for robot manipulation.",
        )
        paper_radar.classify_and_score(paper, [])
        self.assertEqual(paper.track, "vla")

    def test_penalizes_unrelated_world_model(self):
        paper = self.paper(
            title="Earth System World Model for Climate Forecasting",
            abstract="A weather forecasting model for terrestrial ecosystems.",
            categories=["cs.LG"],
            sources=["daily_arxiv"],
        )
        paper_radar.classify_and_score(paper, ["earth system", "weather forecasting"])
        self.assertLess(paper.relevance_score, 5)

    def test_render_escapes_untrusted_metadata(self):
        paper = self.paper(title='<script>alert("x")</script>', summary_zh="A & B")
        rendered = paper_radar.render_article(paper, featured=True)
        self.assertNotIn("<script>", rendered)
        self.assertIn("&lt;script&gt;", rendered)
        self.assertIn("A &amp; B", rendered)

    def test_recent_uses_published_or_updated_date(self):
        paper = self.paper(
            published="2025-01-01T00:00:00Z",
            updated="2026-09-08T00:00:00Z",
        )
        self.assertTrue(paper_radar.recent_enough(paper, dt.date(2026, 9, 1)))

    def test_detects_existing_date_section(self):
        markdown = '<h2 id="papers-2026-09-10" class="paper-day-heading">2026-09-10</h2>'
        self.assertTrue(
            paper_radar.has_date_section(markdown, dt.date(2026, 9, 10))
        )
        self.assertFalse(
            paper_radar.has_date_section(markdown, dt.date(2026, 9, 11))
        )

    def test_run_date_uses_beijing_calendar_day(self):
        utc_time = dt.datetime(2026, 9, 10, 16, 30, tzinfo=dt.timezone.utc)
        self.assertEqual(
            paper_radar.current_run_date(utc_time),
            dt.date(2026, 9, 11),
        )


if __name__ == "__main__":
    unittest.main()
