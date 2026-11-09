"""Compare preprocessing with windows and probabilities saved by the notebook."""
from pathlib import Path
import numpy as np
import wfdb
from app.ml.preprocessing import filter_continuous, extract_windows

ROOT = Path(__file__).resolve().parents[2] / "results" / "mitdb"
REF = Path(__file__).resolve().parents[2] / "results" / "test_examples_for_xai.npz"

def main() -> None:
    with np.load(REF, allow_pickle=False) as z:
        wanted_x, records, positions, reference_prob = z["X"], z["record"], z["position"], z["prob"]
    reconstructed = np.empty_like(wanted_x)
    for record in np.unique(records):
        idx = np.flatnonzero(records == record)
        rec = wfdb.rdrecord(str(ROOT / str(record)), physical=True)
        lead_i = rec.sig_name.index("MLII") if "MLII" in rec.sig_name else 0
        filt = filter_continuous(rec.p_signal[:, lead_i], float(rec.fs))
        windows = extract_windows(filt, positions[idx]).windows
        if len(windows) != len(idx):
            raise RuntimeError(f"window count mismatch for record {record}")
        reconstructed[idx] = windows
    delta = np.abs(reconstructed.astype(np.float64) - wanted_x.astype(np.float64))
    print({"matching_windows": len(wanted_x), "max_abs_input_difference": float(delta.max()),
           "mean_abs_input_difference": float(delta.mean()), "saved_probability_rows": len(reference_prob),
           "prediction_comparison": "unavailable until TensorFlow/model inference can run"})

if __name__ == "__main__": main()
