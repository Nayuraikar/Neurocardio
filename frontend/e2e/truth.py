"""Ground truth that needs Python: counts taken straight from the SQLite file and the WFDB annotations.

Deliberately does not import the app's services, so it cannot repeat their mistakes.
"""
import json
import sqlite3
import sys
from pathlib import Path

import wfdb

root, database, mode = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
config = json.loads((root / "backend/models/preprocessing_config.json").read_text())
if mode == "db":
    con = sqlite3.connect(database)
    one = lambda q: con.execute(q).fetchone()[0]
    # flagged windows whose latest review (if any) is not 'reviewed'
    pending = """
      SELECT b.analysis_id AS a, count(*) AS n FROM beats b JOIN predictions p ON p.beat_id = b.id
      WHERE p.review_flag = 1 AND coalesce((SELECT r.status FROM reviews r WHERE r.beat_id = b.id
            ORDER BY r.created_at DESC LIMIT 1), '') <> 'reviewed' GROUP BY b.analysis_id"""
    needing = f"""
      SELECT x.n FROM ({pending}) x JOIN analyses an ON an.id = x.a WHERE an.status = 'completed'
      AND coalesce((SELECT s.status FROM analysis_reviews s WHERE s.analysis_id = x.a
            ORDER BY s.created_at DESC LIMIT 1), '') <> 'reviewed'"""
    rows = [r[0] for r in con.execute(needing)]
    print(json.dumps({"recordings": one("SELECT count(*) FROM ecg_records"),
                      "completed_analyses": one("SELECT count(*) FROM analyses WHERE status = 'completed'"),
                      "review_required": len(rows), "pending_review_windows": sum(rows)}))
else:
    ann = wfdb.rdann(str(root / "results/mitdb/100"), "atr")
    mapped = [(int(s), config["aami_mapping"][y]) for s, y in zip(ann.sample, ann.symbol) if y in config["aami_mapping"]]
    counts = {c: sum(1 for _, k in mapped if k == c) for c in "NSVF"}
    print(json.dumps({"counts": counts, "complete_N": sum(1 for s, k in mapped if k == "N" and 360 <= s <= 650000 - 360)}))
