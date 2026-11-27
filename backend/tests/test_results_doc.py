import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend" / "tools"))

from build_results_doc import build  # noqa: E402


def test_results_doc_matches_the_committed_reports():
    """docs/RESULTS.md is generated; a hand edit or a changed report makes it stale."""
    assert (ROOT / "docs" / "RESULTS.md").read_text() == build()
