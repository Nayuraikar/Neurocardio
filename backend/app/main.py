from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.api.dataset import router as dataset_router
from app.api.demo import router as demo_router
from app.api.recordings import router as recordings_router
from app.api.system import router as system_router
from app.services.overview_service import fail_interrupted_analyses
from app.core.config import (ARTIFACTS_DIR, CONFIG_PATH, DATA_DIR, DATABASE_URL, LOCAL_FRONTEND_ORIGIN,
                             MAX_UPLOAD_BYTES, MODEL_PATH, RESULTS_DIR)
from app.core.errors import ServiceError
from app.database.base import Base
from app.database.session import make_engine, make_session_factory
from app.models import *  # noqa: F403 - register SQLAlchemy entities before create_all
from app.ml.inference import ECGClassifier, sha256_file


def create_app(*, database_url: str = DATABASE_URL, data_dir: Path = DATA_DIR,
               model_path: Path = MODEL_PATH, config_path: Path = CONFIG_PATH,
               results_dir: Path = RESULTS_DIR, artifacts_dir: Path = ARTIFACTS_DIR, max_upload_bytes: int = MAX_UPLOAD_BYTES,
               classifier=None, initialize_model: bool = True) -> FastAPI:
    engine = make_engine(database_url)
    SessionLocal = make_session_factory(engine)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        Base.metadata.create_all(bind=engine)
        with engine.begin() as conn:
            try:
                res = conn.exec_driver_sql("PRAGMA table_info(explanations)").fetchall()
                cols = [r[1] for r in res]
                if cols and "cache_key" not in cols:
                    conn.exec_driver_sql("ALTER TABLE explanations ADD COLUMN cache_key VARCHAR(64)")
                    conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS ix_explanations_cache_key ON explanations (cache_key)")
                conn.exec_driver_sql("DELETE FROM explanations WHERE cache_key IS NULL")
            except Exception:
                pass
        app.state.classifier = classifier if classifier is not None else (ECGClassifier(model_path) if initialize_model else None)
        app.state.config = json.loads(Path(config_path).read_text())
        app.state.config_sha256 = sha256_file(Path(config_path))
        app.state.data_dir = Path(data_dir)
        app.state.results_dir = Path(results_dir)
        app.state.artifacts_dir = Path(artifacts_dir)
        app.state.max_upload_bytes = int(max_upload_bytes)
        app.state.SessionLocal = SessionLocal
        with SessionLocal() as db:
            fail_interrupted_analyses(db)
        yield
        engine.dispose()

    app = FastAPI(title="NeuroCardio Research API", version="1.0.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=[LOCAL_FRONTEND_ORIGIN], allow_credentials=False,
                       allow_methods=["GET", "POST", "DELETE", "OPTIONS"], allow_headers=["*"])
    # The UI uses /api; the original routes stay reachable under /api/v1.
    app.include_router(router, prefix="/api")
    app.include_router(system_router, prefix="/api")
    app.include_router(recordings_router, prefix="/api")
    app.include_router(dataset_router, prefix="/api")
    app.include_router(demo_router, prefix="/api")
    app.include_router(router, prefix="/api/v1", include_in_schema=False)

    @app.exception_handler(ServiceError)
    async def service_error_handler(request: Request, exc: ServiceError):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    return app


app = create_app()
