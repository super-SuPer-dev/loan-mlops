"""รัน test case (ไฟล์ CSV) ผ่าน API จริง — ใช้วันนำเสนอเมื่ออาจารย์ให้ข้อมูลมาทดสอบ

    uv run python scripts/run_test_cases.py --file <ไฟล์ของอาจารย์>.csv
    uv run python scripts/run_test_cases.py --file data/production/week_1_normal.csv
    uv run python scripts/run_test_cases.py --file data/production/bad_data.csv

ทำ 3 อย่าง
  1. ตรวจทั้งไฟล์กับ schema (แบบ batch) แล้วพิมพ์ปัญหาที่เจอ — ข้อมูลเสียจะเห็นตรงนี้ก่อน
  2. ส่งทีละแถวเข้า POST /predict แสดงผลแต่ละแถว: 200 = ทำนายได้, 400 = ข้อมูลผิด (API ปฏิเสธถูกต้อง)
  3. ถ้าไฟล์มีคอลัมน์ Loan_Approved (เฉลย) คำนวณ ROC-AUC และ accuracy ของแถวที่ทำนายได้
ผลทั้งหมดเขียนลง reports/test_cases.json
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import requests
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src import features as F  # noqa: E402
from src import validation as V  # noqa: E402


def http_post(url: str):
    """คืนฟังก์ชันที่ส่ง payload เข้า /predict แล้วคืน (status, body) — แยกออกมาเพื่อให้เทสต์สลับเป็น Flask test client ได้"""
    session = requests.Session()

    def post(payload: dict) -> tuple[int, dict]:
        try:
            r = session.post(f"{url}/predict", json=payload, timeout=10)
            return r.status_code, r.json()
        except requests.RequestException as e:
            return 599, {"error": f"ติดต่อ API ไม่ได้: {e}"}
    return post


def check_schema(df: pd.DataFrame) -> list[str]:
    """ตรวจทั้งไฟล์กับ schema ของระบบ (ถ้ายังไม่มี schema เพราะยังไม่เคยเทรน ให้ข้าม)"""
    if not C.SCHEMA_PATH.exists():
        return ["(ข้าม: ยังไม่มี include/schema.json ให้รัน loan_training_pipeline ก่อน)"]
    return V.validate(df, V.load_schema(C.SCHEMA_PATH))


def run(df: pd.DataFrame, post) -> dict:
    """ส่งทุกแถวเข้า API แล้วสรุปผล — post(payload) -> (status, body)"""
    has_label = F.TARGET in df.columns
    # to_json → loads ทำให้ NaN กลายเป็น null (JSON ส่ง NaN ไม่ได้)
    rows = json.loads(df.drop(columns=[F.TARGET], errors="ignore").to_json(orient="records"))
    results = []
    for i, payload in enumerate(rows):
        status, body = post(payload)
        results.append({
            "row": i, "status": status,
            "decision": body.get("decision"), "probability_approve": body.get("probability_approve"),
            "approved": body.get("approved"),
            "errors": body.get("details") or ([body["error"]] if status != 200 and "error" in body else []),
        })

    ok = [r for r in results if r["status"] == 200]
    summary = {
        "n_rows": len(results),
        "status_counts": dict(Counter(str(r["status"]) for r in results)),
        "decision_counts": dict(Counter(r["decision"] for r in ok)),
    }
    if has_label and ok:
        y = F.encode_target(df[F.TARGET]).to_numpy()
        idx = [r["row"] for r in ok]
        y_ok = y[idx]
        summary["accuracy"] = round(float(accuracy_score(y_ok, [r["approved"] for r in ok])), 4)
        if len(set(y_ok)) == 2:           # AUC ต้องมีทั้งสองคลาส
            summary["roc_auc"] = round(float(roc_auc_score(y_ok, [r["probability_approve"] for r in ok])), 4)
    return {"summary": summary, "rows": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="ไฟล์ CSV ของ test case")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--show", type=int, default=20, help="จำนวนแถวที่พิมพ์ให้ดู")
    args = ap.parse_args()

    df = pd.read_csv(args.file)
    print(f"ไฟล์: {args.file}  ({len(df):,} แถว, {df.shape[1]} คอลัมน์)")

    problems = check_schema(df)
    print("\n[1] ตรวจทั้งไฟล์กับ schema:", "ผ่าน ✅" if not problems else f"พบ {len(problems)} ปัญหา ❌")
    for p in problems:
        print("   -", p)

    report = run(df, http_post(args.url))
    print(f"\n[2] ผลต่อแถว (แสดง {min(args.show, len(df))} แถวแรก)")
    for r in report["rows"][: args.show]:
        if r["status"] == 200:
            print(f"   แถว {r['row']:>4}: 200  {r['decision']:<14} p={r['probability_approve']}")
        else:
            print(f"   แถว {r['row']:>4}: {r['status']}  {'; '.join(r['errors'])[:110]}")

    print("\n[3] สรุป:", json.dumps(report["summary"], ensure_ascii=False))
    out = C.ROOT / "reports" / "test_cases.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"file": args.file, "schema_problems": problems, **report},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print("บันทึกผลไว้ที่", out)


if __name__ == "__main__":
    main()
