from __future__ import annotations

from io import BytesIO
from pathlib import Path
import json
import math
import uuid

import numpy as np
import pandas as pd
import wfdb
from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.core.errors import ServiceError
from app.core.config import ANNOTATION_ALIGNMENT_MIN, ANNOTATION_ALIGNMENT_TOLERANCE_MS, CONFIG_PATH
from app.ml.preprocessing import FS_HZ, filter_continuous, validate_signal
from app.ml.rpeak_detection import annotation_alignment, detect_r_peaks
from app.models import ECGRecord


def _preview(signal: np.ndarray, fs: float, max_points: int = 1200) -> list[dict]:
    stride = max(1, int(math.ceil(len(signal) / max_points)))
    idx = np.arange(0, len(signal), stride, dtype=np.int64)
    return [{"sample_index": int(i), "time_seconds": float(i / fs), "amplitude": float(signal[i])} for i in idx]


def _validate_recording(signal: np.ndarray, sampling_rate: float) -> np.ndarray:
    try:
        x = validate_signal(signal, sampling_rate)
    except ValueError as exc:
        raise ServiceError(422, str(exc)) from exc
    if len(x) < 720:
        raise ServiceError(422, "recording is shorter than 720 samples (2 seconds at 360 Hz)")
    if float(np.ptp(x)) == 0.0:
        raise ServiceError(422, "flatline signal is not accepted")
    filtered = filter_continuous(x, sampling_rate)
    peaks = detect_r_peaks(filtered, sampling_rate)
    if len(peaks) == 0:
        raise ServiceError(422, "no R peaks detected in the selected lead")
    return peaks


async def upload_recording(files: list[UploadFile], sampling_rate: float | None, lead: str | None,
                           session: Session, data_root: Path, max_upload_bytes: int) -> tuple[ECGRecord, list[dict]]:
    if not files:
        raise ServiceError(400, "at least one file is required")
    payloads = []
    total_size = 0
    for file in files:
        name = Path(file.filename or "").name
        if not name:
            raise ServiceError(400, "uploaded file has no filename")
        content = await file.read(max_upload_bytes + 1)
        total_size += len(content)
        payloads.append((name, content))
        if total_size > max_upload_bytes:
            raise ServiceError(413, f"upload exceeds configured {max_upload_bytes}-byte limit")
    names = [n.lower() for n, _ in payloads]
    csv_files = [x for x in payloads if x[0].lower().endswith(".csv")]
    hea_files = [x for x in payloads if x[0].lower().endswith(".hea")]
    if csv_files:
        if len(payloads) != 1 or len(csv_files) != 1:
            raise ServiceError(400, "CSV upload must contain exactly one .csv file")
        if sampling_rate is None:
            raise ServiceError(422, "sampling_rate is required for CSV uploads")
        if not lead:
            raise ServiceError(422, "lead column selection is required for CSV uploads")
        try:
            frame = pd.read_csv(BytesIO(csv_files[0][1]))
        except Exception as exc:
            raise ServiceError(422, f"malformed CSV: {exc}") from exc
        if lead not in frame.columns:
            raise ServiceError(422, f"lead column {lead!r} was not found in CSV")
        try:
            values = pd.to_numeric(frame[lead], errors="raise").to_numpy(dtype=np.float64)
        except Exception as exc:
            raise ServiceError(422, f"lead column {lead!r} contains non-numeric values") from exc
        if not np.isfinite(values).all():
            raise ServiceError(422, "signal contains NaN or infinite values")
        _validate_recording(values, float(sampling_rate))
        rid = str(uuid.uuid4())
        folder = data_root / "records" / rid
        folder.mkdir(parents=True, exist_ok=True)
        stored = folder / "signal.npy"
        np.save(stored, values.astype(np.float32))
        record = ECGRecord(id=rid, filename=csv_files[0][0], format="csv", sampling_rate=float(sampling_rate),
                           lead=lead, duration_seconds=len(values) / float(sampling_rate), sample_count=len(values),
                           signal_path=str(stored), metadata_json={"columns": list(frame.columns), "upload_bytes": total_size})
    else:
        if not hea_files:
            raise ServiceError(415, "unsupported upload; provide a CSV or WFDB .hea with matching .dat")
        allowed_suffixes = {".hea", ".dat", ".atr"}
        unexpected = [(n, data) for n, data in payloads if Path(n).suffix.lower() not in allowed_suffixes]
        if unexpected:
            raise ServiceError(415, "WFDB upload accepts only .hea, .dat and optional .atr files: " +
                               ", ".join(n for n, _ in unexpected))
        hea_files = [(n, data) for n, data in payloads if Path(n).suffix.lower() == ".hea"]
        dat_files = [(n, data) for n, data in payloads if Path(n).suffix.lower() == ".dat"]
        atr_files = [(n, data) for n, data in payloads if Path(n).suffix.lower() == ".atr"]
        if len(hea_files) != 1:
            raise ServiceError(400, "WFDB upload must contain exactly one .hea header")
        if len(dat_files) != 1:
            missing = Path(hea_files[0][0]).with_suffix(".dat").name
            raise ServiceError(422, f"WFDB header {hea_files[0][0]} requires exactly one matching .dat file (expected {missing})")
        if len(atr_files) > 1:
            raise ServiceError(422, "WFDB upload accepts at most one optional .atr annotation file: " +
                               ", ".join(n for n, _ in atr_files))
        hea_name, hea_data = hea_files[0]
        stem = Path(hea_name).stem
        named_parts = [(hea_name, Path(hea_name).stem), (dat_files[0][0], Path(dat_files[0][0]).stem)]
        if atr_files:
            named_parts.append((atr_files[0][0], Path(atr_files[0][0]).stem))
        mismatched = [(name, part_stem) for name, part_stem in named_parts if part_stem != stem]
        if mismatched:
            details = ", ".join(f"{name} (base {part_stem!r})" for name, part_stem in named_parts)
            raise ServiceError(422, f"WFDB companion filenames must share one base name; mismatched files: {details}")
        try:
            header_line = next(line.strip() for line in hea_data.decode("ascii").splitlines() if line.strip())
            header_record_name = header_line.split()[0]
        except (UnicodeDecodeError, StopIteration, IndexError) as exc:
            raise ServiceError(422, f"invalid WFDB header {hea_name}: missing record name") from exc
        if header_record_name != stem:
            raise ServiceError(422, f"WFDB header {hea_name} declares record {header_record_name!r}, but its filename base is {stem!r}")
        rid = str(uuid.uuid4())
        folder = data_root / "records" / rid
        folder.mkdir(parents=True, exist_ok=True)
        for name, content in payloads:
            (folder / Path(name).name).write_bytes(content)
        base = folder / stem
        try:
            rec = wfdb.rdrecord(str(base), physical=True)
        except Exception as exc:
            raise ServiceError(422, f"invalid WFDB header or signal file: {exc}") from exc
        if str(rec.record_name) != header_record_name:
            raise ServiceError(422, f"WFDB header record name {header_record_name!r} does not match loaded record {rec.record_name!r}")
        if not np.isclose(float(rec.fs), FS_HZ, rtol=0.0, atol=1e-9):
            raise ServiceError(422, f"WFDB sampling rate {rec.fs:g} Hz in {hea_name} does not match required {FS_HZ:g} Hz")
        selected_lead = lead or ("MLII" if "MLII" in rec.sig_name else rec.sig_name[0])
        if selected_lead not in rec.sig_name:
            raise ServiceError(422, f"lead {selected_lead!r} is not present; available leads: {rec.sig_name}")
        if sampling_rate is not None and not np.isclose(sampling_rate, rec.fs):
            raise ServiceError(422, f"supplied sampling_rate {sampling_rate} disagrees with WFDB header {rec.fs}")
        signal = rec.p_signal[:, rec.sig_name.index(selected_lead)].astype(np.float64)
        detected_peaks = _validate_recording(signal, float(rec.fs))
        alignment = None
        atr = atr_files[0] if atr_files else None
        if atr:
            try:
                ann = wfdb.rdann(str(base), Path(atr[0]).suffix.lstrip("."))
            except Exception as exc:
                raise ServiceError(422, f"invalid WFDB annotation file {atr[0]}: {exc}") from exc
            mapping = json.loads(CONFIG_PATH.read_text()).get("aami_mapping", {})
            annotation_positions = np.asarray([p for p, symbol in zip(ann.sample, ann.symbol) if symbol in mapping], dtype=np.int64)
            if len(annotation_positions):
                alignment = annotation_alignment(annotation_positions, detected_peaks, float(rec.fs),
                                                 ANNOTATION_ALIGNMENT_TOLERANCE_MS)
                if alignment["matched_fraction"] < ANNOTATION_ALIGNMENT_MIN:
                    raise ServiceError(422, f"annotation alignment check failed for {hea_name} + {atr[0]}: "
                                       f"{alignment['matched_count']}/{alignment['annotation_count']} AAMI peaks "
                                       f"({alignment['matched_fraction']:.1%}) matched XQRS within "
                                       f"{ANNOTATION_ALIGNMENT_TOLERANCE_MS:g} ms; required "
                                       f"{ANNOTATION_ALIGNMENT_MIN:.0%}. The annotation may belong to a different recording.")
        stored = folder / "signal.npy"
        np.save(stored, signal.astype(np.float32))
        annotation_path = str(folder / Path(atr[0]).name) if atr else None
        record = ECGRecord(id=rid, filename=hea_name, format="wfdb", sampling_rate=float(rec.fs),
                           lead=selected_lead, duration_seconds=len(signal) / float(rec.fs), sample_count=len(signal),
                           signal_path=str(stored), wfdb_base_path=str(base), annotation_path=annotation_path,
                           metadata_json={"signals": rec.sig_name, "units": rec.units, "header": hea_name,
                                          "has_annotation": bool(atr), "annotation_alignment": alignment,
                                          "annotation_alignment_min_fraction": ANNOTATION_ALIGNMENT_MIN,
                                          "annotation_alignment_tolerance_ms": ANNOTATION_ALIGNMENT_TOLERANCE_MS,
                                          "upload_bytes": total_size})
    session.add(record)
    session.commit()
    session.refresh(record)
    signal = np.load(record.signal_path, mmap_mode="r")
    return record, _preview(signal, record.sampling_rate)
