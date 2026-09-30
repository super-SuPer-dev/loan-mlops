"""Schema และการตรวจความผิดปกติของข้อมูล — ต่อยอดจาก schema.json ของ Lab 11 (ExampleValidator)

ใช้สองที่:
  1. ใน pipeline ก่อนเทรน  → เจอข้อมูลเสียต้อง "หยุด" (raise) ไม่ใช่เทรนต่อ
  2. ใน API ก่อนทำนาย      → คำขอที่ผิด schema ตอบกลับ 400 แทนการเดามั่ว
"""
import json
from pathlib import Path

import pandas as pd

from src.features import CATEGORICAL, ID_COLS, NUMERIC

# กฎจากความรู้โดเมน (ไม่ได้เรียนจากข้อมูล) — ผิดเมื่อไรคือผิดแน่นอน ไม่มีการเผื่อขอบ
DOMAIN_RULES = {
    "Age": (18, 100),
    "Credit_Score": (300, 850),
    "Risk_Score": (0, 100),
    "Debt_To_Income_Ratio": (0, 100),
    "Loan_Term_Months": (1, 480),
}
NON_NEGATIVE = [
    "Job_Experience_Years", "Annual_Income", "Monthly_Income", "Credit_History_Years",
    "Existing_Loans", "Late_Payments_Last_2_Years", "Loan_Amount", "Savings_Balance",
    "Checking_Balance", "Total_Assets", "Total_Liabilities", "Monthly_Debt_Payment",
    "Dependents", "Bank_Account_Age_Years",
]
RANGE_TOLERANCE = 0.2      # ช่วงที่เรียนจาก train เผื่อได้ 20% ของความกว้าง (เหมือน Lab 11)
MAX_MISSING_RATIO = 0.05   # ทั้ง batch ขาดเกิน 5% ในคอลัมน์ใด = ผิดปกติ


class DataValidationError(Exception):
    """โยนเมื่อข้อมูลไม่ผ่าน schema — ทำให้ task ใน Airflow ล้ม (สีแดง) และ pipeline หยุด"""

    def __init__(self, anomalies):
        self.anomalies = anomalies
        super().__init__("ข้อมูลไม่ผ่านการตรวจ:\n  - " + "\n  - ".join(anomalies))


def build_schema(train_df: pd.DataFrame, target: str | None = None) -> dict:
    """สร้าง schema จากชุด train (SchemaGen) — สร้างครั้งเดียวแล้วเก็บเป็นไฟล์ ไม่สร้างใหม่ทุกรอบ

    "required" = คอลัมน์ที่ลูกค้ากรอกตอนยื่นคำขอ (ฟีเจอร์ของโมเดล) ต้องมีเสมอ
    คอลัมน์อื่น เช่น Approved_Amount ที่รู้หลังตัดสินใจ ไม่บังคับ แต่ถ้าส่งมาก็ถูกตรวจช่วงค่าด้วย
    """
    schema = {"numeric": {}, "categorical": {}, "target": None,
              "required": [c for c in NUMERIC + CATEGORICAL if c in train_df.columns]}
    for col in train_df.columns:
        if col in ID_COLS:
            continue                          # เลขคำขอเพิ่มขึ้นเรื่อย ๆ ไม่ควรมีกฎช่วงค่า (ตรวจแค่ห้ามซ้ำ)
        if col == target:
            schema["target"] = {"name": col, "domain": sorted(train_df[col].dropna().unique().tolist())}
        elif pd.api.types.is_numeric_dtype(train_df[col]):
            schema["numeric"][col] = {"min": float(train_df[col].min()), "max": float(train_df[col].max())}
        else:
            schema["categorical"][col] = {"domain": sorted(train_df[col].dropna().astype(str).unique().tolist())}
    return schema


def save_schema(schema: dict, path: Path) -> None:
    Path(path).write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")


def load_schema(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate(df: pd.DataFrame, schema: dict, batch: bool = True) -> list[str]:
    """คืนรายการความผิดปกติ (list ว่าง = ผ่าน)

    batch=True  ตรวจแบบทั้งชุด (มีกฎสัดส่วนค่าว่าง) ใช้ใน pipeline
    batch=False ตรวจคำขอทีละแถว ใช้ใน API — ค่าว่างรายตัวยอมรับได้เพราะ imputer เติมให้
    """
    anomalies = []
    required = schema.get("required", list(schema["numeric"]) + list(schema["categorical"]))
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        anomalies.append(f"ขาดคอลัมน์ที่ schema กำหนด: {missing_cols}")

    # คำนวณสถิติของทุกคอลัมน์ตัวเลขทีเดียว แล้วค่อยวนตรวจด้วยตัวเลขธรรมดา (เร็วพอสำหรับ API)
    num_cols = [c for c in schema["numeric"] if c in df.columns]
    raw = df[num_cols]
    num = raw.apply(pd.to_numeric, errors="coerce")
    bad_types = (num.isna() & raw.notna()).sum().to_dict()
    missing = num.isna().mean().to_dict()
    lows, highs = num.min().to_dict(), num.max().to_dict()

    for col in num_cols:
        rule = schema["numeric"][col]
        if bad_types[col]:
            anomalies.append(f"{col}: มีค่าที่ไม่ใช่ตัวเลข {bad_types[col]} แถว")
        if batch and missing[col] > MAX_MISSING_RATIO:
            anomalies.append(f"{col}: ค่าว่าง {missing[col]:.1%} เกินเกณฑ์ {MAX_MISSING_RATIO:.0%}")
        lo, hi = lows[col], highs[col]
        if pd.isna(lo):
            continue
        if col in DOMAIN_RULES:
            dlo, dhi = DOMAIN_RULES[col]
            if lo < dlo or hi > dhi:
                anomalies.append(f"{col}: ผิดกฎโดเมน ต้องอยู่ใน [{dlo}, {dhi}] แต่พบ {lo:g} ถึง {hi:g}")
                continue
        if col in NON_NEGATIVE and lo < 0:
            anomalies.append(f"{col}: พบค่าติดลบ {lo:g} ทั้งที่ห้ามติดลบ")
            continue
        span = (rule["max"] - rule["min"]) or 1
        if lo < rule["min"] - RANGE_TOLERANCE * span or hi > rule["max"] + RANGE_TOLERANCE * span:
            anomalies.append(f"{col}: ค่าอยู่นอกช่วงที่คาด ({lo:g} ถึง {hi:g})")

    for col, rule in schema["categorical"].items():
        if col not in df.columns:
            continue
        if batch and df[col].isna().mean() > MAX_MISSING_RATIO:
            anomalies.append(f"{col}: ค่าว่าง {df[col].isna().mean():.1%} เกินเกณฑ์ {MAX_MISSING_RATIO:.0%}")
        unseen = set(df[col].dropna().astype(str).unique()) - set(rule["domain"])
        if unseen:
            anomalies.append(f"{col}: พบหมวดที่ไม่เคยเห็น {sorted(unseen)}")

    # กฎความสอดคล้องข้ามคอลัมน์: รายได้ต่อเดือนต้องเท่ากับรายได้ต่อปี / 12
    if {"Monthly_Income", "Annual_Income"} <= set(df.columns):
        diff = (pd.to_numeric(df["Monthly_Income"], errors="coerce")
                - pd.to_numeric(df["Annual_Income"], errors="coerce") / 12).abs()
        n_bad = int((diff > 1.0).sum())
        if n_bad:
            anomalies.append(f"Monthly_Income ไม่เท่ากับ Annual_Income/12 จำนวน {n_bad} แถว")

    tgt = schema.get("target")
    if tgt and tgt["name"] in df.columns:
        unseen = set(df[tgt["name"]].dropna().unique()) - set(tgt["domain"])
        if unseen:
            anomalies.append(f"{tgt['name']}: label แปลกปลอม {sorted(unseen)}")

    if batch and "Application_ID" in df.columns and df["Application_ID"].duplicated().any():
        anomalies.append(f"Application_ID ซ้ำ {int(df['Application_ID'].duplicated().sum())} แถว")
    return anomalies


def validate_or_raise(df: pd.DataFrame, schema: dict, batch: bool = True) -> None:
    anomalies = validate(df, schema, batch=batch)
    if anomalies:
        raise DataValidationError(anomalies)
