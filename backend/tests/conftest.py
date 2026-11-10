from pathlib import Path
import sys

import pytest

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.ml.inference import ECGClassifier


@pytest.fixture(scope="session")
def model():
    return ECGClassifier()


def pytest_collection_modifyitems(config, items):
    import inspect
    ref = BACKEND.parent / "results" / "test_examples_for_xai.npz"
    if ref.exists():
        return
    skip = pytest.mark.skip(reason="results/test_examples_for_xai.npz not in repo (too large for GitHub)")
    for item in items:
        try:
            if "test_examples_for_xai" in inspect.getsource(item.function):
                item.add_marker(skip)
        except (AttributeError, OSError, TypeError):
            pass
