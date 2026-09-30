"""ด่านตรวจใน CI ที่ใช้ "ข้อมูลจริง" (ต่างจาก pytest ที่ใช้ข้อมูลสังเคราะห์ขนาดเล็ก)

    uv run python scripts/ci_checks.py data    # ด่านความถูกต้องของข้อมูล
    uv run python scripts/ci_checks.py model   # ด่านเกณฑ์คุณภาพของโมเดล

ทั้งสองคำสั่ง exit code = 1 เมื่อไม่ผ่าน → GitHub Actions ขึ้นสีแดงและบล็อกการ merge
ผลทุกอย่างเขียนลงโฟลเดอร์ .ci/ แยกจากระบบที่รันอยู่ และสรุปเป็นตารางในหน้า Actions (GITHUB_STEP_SUMMARY)
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src import steps  # noqa: E402
from src import validation as V  # noqa: E402
from src.data_prep import prepare_all  # noqa: E402

WORK = C.ROOT / ".ci"


def isolate() -> None:
    """ให้ artifact, schema และ MLflow ของ CI ไปอยู่ใน .ci/ ไม่ยุ่งกับของระบบจริง"""
    C.RUNS = WORK / "runs"
    C.SCHEMA_PATH = WORK / "schema.json"
    C.ALERTS_LOG = WORK / "alerts.jsonl"
    C.MLFLOW_TRACKING_URI = f"sqlite:///{(WORK / 'mlflow.db').as_posix()}"
    WORK.mkdir(exist_ok=True)


def summary(title: str, rows: list[tuple], ok: bool) -> None:
    lines = [f"### {'✅' if ok else '❌'} {title}", "", "| รายการ | ผล |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rows]
    text = "\n".join(lines) + "\n"
    print(text)
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(text)


def check_data() -> bool:
    info = prepare_all()                                   # ดาวน์โหลดจาก Kaggle + แบ่งข้อมูล (seed คงที่)
    ex = steps.example_gen("ci_data")
    schema = V.load_schema(steps.schema_gen(ex))
    rows, ok = [("จำนวนแถว historical", info["historical"]), ("data md5", ex["data_md5"])], True

    result = steps.example_validator(ex, str(C.SCHEMA_PATH))
    rows.append(("train/val/test ผ่าน schema", "ผ่าน" if result["ok"] else result["anomalies"][:3]))
    ok &= result["ok"]
    for name in ["week_1_normal.csv", "week_2_data_drift.csv", "week_3_concept_drift.csv"]:
        problems = V.validate(pd.read_csv(C.PRODUCTION / name), schema)
        rows.append((f"{name} ผ่าน schema", "ผ่าน" if not problems else problems[:3]))
        ok &= not problems
    # ข้อมูลเสียต้อง "ไม่ผ่าน" — ถ้าผ่าน แปลว่าด่านตรวจเสื่อม (เช่น มีคนไปลบกฎทิ้ง)
    caught = V.validate(pd.read_csv(C.PRODUCTION / "bad_data.csv"), schema)
    rows.append(("bad_data.csv ถูกจับได้", f"{len(caught)} ปัญหา" if caught else "ไม่ถูกจับ!"))
    ok &= len(caught) >= 5
    summary("Data validation (ข้อมูลจริง)", rows, ok)
    return ok


def check_model() -> bool:
    prepare_all()
    ex = steps.example_gen("ci_model")
    schema_path = steps.schema_gen(ex)
    best = steps.select_best([steps.trainer(c, ex, schema_path, "ci") for c in ("logreg", "hist_gboost")])
    m = steps.evaluator(ex, best)
    fair = steps.fairness_audit(ex, best)
    rows = [("โมเดลที่เลือก", best["candidate"]), ("threshold", best["threshold"]),
            (f"ROC-AUC (≥ {C.MIN_ROC_AUC})", f"{m['roc_auc']:.4f}"),
            (f"Recall ปฏิเสธ (≥ {C.MIN_RECALL_REJECTED})", f"{m['recall_rejected']:.4f}"),
            (f"p95 latency ms (≤ {C.MAX_P95_LATENCY_MS})", f"{m['p95_latency_ms']:.2f}"),
            (f"ขนาดโมเดล MB (≤ {C.MAX_MODEL_SIZE_MB})", f"{m['model_size_mb']:.3f}"),
            (f"Fairness: ส่วนต่างอัตราอนุมัติ (≤ {C.MAX_APPROVAL_GAP})", fair["gap"])]
    rows += [(f"ด่าน {k}", "ผ่าน" if v else "ไม่ผ่าน") for k, v in m["checks"].items()]
    ok = m["blessed"] and fair["ok"]
    summary("Model quality gate (ข้อมูลจริง)", rows, ok)
    (WORK / "model_gate.json").write_text(json.dumps({"metrics": m, "fairness": fair}, indent=2, default=str))
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("check", choices=["data", "model"])
    isolate()
    ok = check_data() if ap.parse_args().check == "data" else check_model()
    sys.exit(0 if ok else 1)
