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
