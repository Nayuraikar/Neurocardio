from pathlib import Path
import os

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = BACKEND_DIR.parent
DATA_DIR = BACKEND_DIR / "data"
MODEL_PATH = BACKEND_DIR / "models" / "neurocardio_final.keras"
CONFIG_PATH = BACKEND_DIR / "models" / "preprocessing_config.json"
RESULTS_DIR = PROJECT_DIR / "results"
DATABASE_URL = f"sqlite:///{(DATA_DIR / 'neurocardio.sqlite3').as_posix()}"
PREPROCESSING_VERSION = "mitdb-360hz-butter4-0.5-40-window720-zscore-eps1e-8-v1"
MAX_UPLOAD_BYTES = 64 * 1024 * 1024
LOCAL_FRONTEND_ORIGIN = "http://localhost:5173"
ANNOTATION_ALIGNMENT_MIN = float(os.getenv("NEUROCARDIO_ANNOTATION_ALIGNMENT_MIN", "0.90"))
ANNOTATION_ALIGNMENT_TOLERANCE_MS = float(os.getenv("NEUROCARDIO_ANNOTATION_ALIGNMENT_TOLERANCE_MS", "150"))
if not 0.0 <= ANNOTATION_ALIGNMENT_MIN <= 1.0:
    raise ValueError("NEUROCARDIO_ANNOTATION_ALIGNMENT_MIN must be between 0 and 1")
if ANNOTATION_ALIGNMENT_TOLERANCE_MS <= 0.0:
    raise ValueError("NEUROCARDIO_ANNOTATION_ALIGNMENT_TOLERANCE_MS must be positive")
