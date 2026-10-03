"""รายงาน data drift แบบ HTML ด้วย Evidently (Lab 10)

KS / Chi-square ใน src/monitoring.py ให้ "ตัวเลข" ไว้ตัดสินใจอัตโนมัติ (แจ้งเตือน / retrain)
รายงานนี้ให้ "ภาพ" ไว้ให้คนดูว่า drift ที่ฟีเจอร์ไหน การกระจายตัวเปลี่ยนไปอย่างไร
จึงเป็นส่วนเสริม ถ้าสร้างไม่สำเร็จ ห้ามทำให้ monitoring DAG ล้ม
"""
from pathlib import Path

import pandas as pd

from src import features as F

REFERENCE_SAMPLE = 5000   # สุ่ม reference มาเท่านี้พอ ทำให้รายงานสร้างเร็ว (ข้อมูลอดีตทั้งหมดมี 24,000 แถว)


def build_drift_report(reference: pd.DataFrame, current: pd.DataFrame, out_path: Path, seed: int = 42) -> dict:
    """สร้างรายงาน Evidently DataDriftPreset เทียบ reference กับ batch ปัจจุบัน แล้วบันทึกเป็น HTML

    คืน dict สรุป: path ของไฟล์, จำนวนและสัดส่วนคอลัมน์ที่ Evidently ตัดสินว่า drift
    """
    # import ข้างในฟังก์ชัน: Evidently ใหญ่ ถ้า import ที่หัวไฟล์ DAG จะ parse ช้าทุกครั้ง
    from evidently import Report
    from evidently.presets import DataDriftPreset

    cols = F.NUMERIC + F.CATEGORICAL          # ดูเฉพาะฟีเจอร์ที่โมเดลใช้จริง
    ref = reference[cols]
    if len(ref) > REFERENCE_SAMPLE:
        ref = ref.sample(REFERENCE_SAMPLE, random_state=seed)

    snapshot = Report([DataDriftPreset()]).run(current_data=current[cols], reference_data=ref)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(out_path))

    # metric แรกของ preset คือ DriftedColumnsCount: {"count": จำนวนคอลัมน์ที่ drift, "share": สัดส่วน}
    summary = next(m["value"] for m in snapshot.dict()["metrics"]
                   if m["metric_name"].startswith("DriftedColumnsCount"))
    return {"html": str(out_path), "drifted_columns": int(summary["count"]),
            "drift_share": round(float(summary["share"]), 3), "n_columns": len(cols)}
