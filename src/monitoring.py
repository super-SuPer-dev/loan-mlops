"""ตรรกะของ monitoring DAG — แยก Data Drift กับ Concept Drift (ต่อยอด monitoring.py ของ Lab 11)

Data drift    : ข้อมูลเข้าเปลี่ยน P(X)  → ตรวจได้ทันทีด้วย KS (ตัวเลข) / Chi-square (หมวดหมู่) ไม่ต้องรอเฉลย
Concept drift : ความสัมพันธ์เปลี่ยน P(y|X) → ต้องรอเฉลย แล้วดูว่า AUC ตกจากตอน deploy เกินเกณฑ์ไหม

นโยบาย retrain
  - concept drift / performance ตก   → retrain ทันทีด้วยข้อมูลล่าสุดที่มีเฉลย
  - data drift อย่างเดียว              → แจ้งเตือนให้ทีมตรวจ ยังไม่ retrain (โมเดลยังแม่นอยู่)
  - ข้อมูลเสีย                          → หยุด ห้าม retrain
"""
import json
import time
from pathlib import Path

import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from src import config as C
from src import features as F
from src import validation as V
from src.steps import load_champion


def drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    drifted = {}
    for col in F.NUMERIC:
        p = stats.ks_2samp(reference[col].dropna(), current[col].dropna()).pvalue
        if p < C.DRIFT_PVALUE:
            drifted[col] = float(p)
    for col in F.CATEGORICAL:
        ref = reference[col].value_counts()
        cur = current[col].value_counts().reindex(ref.index).fillna(0)
        expected = ref / ref.sum() * cur.sum()
        p = stats.chisquare(cur.to_numpy(), expected.to_numpy()).pvalue
        if p < C.DRIFT_PVALUE:
            drifted[col] = float(p)
    n = len(F.NUMERIC) + len(F.CATEGORICAL)
    return {"drift_share": round(len(drifted) / n, 3), "drifted_features": drifted}


def check_batch(batch_file: str) -> dict:
    """ตรวจ batch หนึ่งไฟล์: schema → data drift → ผลโมเดล champion เมื่อเฉลยมาถึง"""
    model, mv = load_champion()
    if model is None or not C.SCHEMA_PATH.exists():
        # ระบบเพิ่งตั้ง ยังไม่มีโมเดลให้เฝ้า → ไม่ถือว่าล้ม และห้าม retrain ด้วย batch นี้ (ข้อมูลน้อยเกินไป)
        return {"batch_file": batch_file, "needs_retrain": False,
                "alerts": ["ยังไม่มีโมเดลให้บริการ ให้รัน loan_training_pipeline ก่อน"]}

    batch = pd.read_csv(C.DATA / batch_file)
    V.validate_or_raise(batch, V.load_schema(C.SCHEMA_PATH))   # ข้อมูลเสีย → โยน error ให้ task ล้ม

    reference = pd.read_csv(C.HISTORICAL_CSV)
    result = {"batch_file": batch_file, "n_rows": len(batch), **drift_report(reference, batch)}
    auc = float(roc_auc_score(F.encode_target(batch[F.TARGET]), model.predict_proba(batch)[:, 1]))
    ref_auc = float(mv.tags.get("test_roc_auc", auc))          # AUC ตอน deploy = เส้นฐาน
    result.update(champion_version=mv.version, roc_auc=round(auc, 4), reference_roc_auc=ref_auc,
                  auc_drop=round(ref_auc - auc, 4))

    result["data_drift"] = result["drift_share"] > C.DRIFT_SHARE_LIMIT
    result["concept_drift"] = result["auc_drop"] > C.AUC_DROP_LIMIT
    result["needs_retrain"] = result["concept_drift"]
    result["alerts"] = ([f"DATA DRIFT {result['drift_share']:.0%} ของฟีเจอร์ (เกณฑ์ {C.DRIFT_SHARE_LIMIT:.0%})"]
                        if result["data_drift"] else []) + \
                       ([f"CONCEPT DRIFT: AUC ตก {result['auc_drop']:.3f} (เกณฑ์ {C.AUC_DROP_LIMIT})"]
                        if result["concept_drift"] else [])
    return result


def write_report(result: dict, out_dir: Path) -> str:
    """เก็บผลไว้ 2 ที่: โฟลเดอร์ของ run นี้ (ย้อนดูได้) และ latest.json (ให้ Prometheus/Grafana แสดงผลล่าสุด)"""
    path = Path(out_dir) / "monitoring.json"
    text = json.dumps({**result, "checked_at": time.time()}, indent=2, ensure_ascii=False)
    path.write_text(text)
    if "roc_auc" in result:                        # เขียน latest เฉพาะรอบที่วัดผลได้จริง
        C.MONITORING_LATEST.parent.mkdir(parents=True, exist_ok=True)
        C.MONITORING_LATEST.write_text(text)
    return str(path)
