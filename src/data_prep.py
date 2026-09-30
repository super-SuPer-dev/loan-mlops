"""เตรียมข้อมูล: ดาวน์โหลดจาก Kaggle → แบ่งข้อมูลอดีต/ข้อมูล production → จำลอง batch รายสัปดาห์

ทำไมต้องกันข้อมูล 20% ไว้ (pool): ชุดข้อมูลนี้ไม่มีคอลัมน์เวลา เราจึงสมมติว่า 80% คือ "ข้อมูลในอดีต" ที่ใช้เทรน
ส่วน 20% คือ "คำขอที่จะเข้ามาในอนาคต" ใช้จำลอง production และ drift (แนวเดียวกับ train/reference/pool ใน Lab 10)
ทุกขั้นใช้ seed ตายตัว รันกี่ครั้งก็ได้ไฟล์เดิม
"""
import io
import zipfile

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from src.config import HISTORICAL_CSV, KAGGLE_URL, POOL_CSV, PRODUCTION, RAW_CSV, SEED
from src.features import TARGET


def download_raw() -> None:
    """ดาวน์โหลด CSV จาก Kaggle (dataset สาธารณะ ไม่ต้องล็อกอิน) ถ้ายังไม่มีไฟล์"""
    if RAW_CSV.exists():
        return
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(KAGGLE_URL, timeout=120)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        RAW_CSV.write_bytes(z.read(RAW_CSV.name))
    print("ดาวน์โหลดแล้ว:", RAW_CSV)


def concept_drift_policy(df: pd.DataFrame) -> pd.DataFrame:
    """นโยบายใหม่ของธนาคาร (ดอกเบี้ยขาขึ้น): ไม่ปล่อยกู้ระยะเกิน 120 เดือน
    ข้อมูลเข้าเหมือนเดิม แต่ label เปลี่ยน = concept drift (P(y|X) เปลี่ยน)"""
    out = df.copy()
    out.loc[out["Loan_Term_Months"] > 120, TARGET] = "Rejected"
    return out


def make_bad_batch(df: pd.DataFrame) -> pd.DataFrame:
    """ข้อมูลเสียหลายแบบที่เจอได้จริง ใช้สาธิตว่า pipeline หยุดและแจ้งเตือน (แนว make_bad_data.py ของ Lab 11)"""
    bad = df.sample(min(2000, len(df)), random_state=SEED).copy()
    bad.loc[bad.index[:30], "Credit_Score"] = 920                  # เกินช่วง 300–850
    bad.loc[bad.index[30:60], "Annual_Income"] *= -1               # รายได้ติดลบ
    bad.loc[bad.index[60:90], "Employment_Status"] = "Freelancer"  # หมวดที่ไม่เคยเห็น
    bad.loc[bad.index[:300], "Savings_Balance"] = np.nan           # ค่าว่าง 15%
    return bad.drop(columns=["Collateral"])                        # คอลัมน์หาย


def prepare_all() -> dict:
    """สร้างไฟล์ข้อมูลทั้งหมดที่ระบบต้องใช้ — idempotent เรียกซ้ำได้

    lineterminator="\\n" บังคับให้ไฟล์เหมือนกันทุกไบต์ทั้งบน Windows และ Linux → md5 (data version) ไม่เปลี่ยน
    """
    download_raw()
    df = pd.read_csv(RAW_CSV)
    hist, pool = train_test_split(df, test_size=0.20, random_state=SEED, stratify=df[TARGET])
    hist.to_csv(HISTORICAL_CSV, index=False, lineterminator="\n")
    pool.to_csv(POOL_CSV, index=False, lineterminator="\n")

    PRODUCTION.mkdir(parents=True, exist_ok=True)
    pool = pool.sample(frac=1, random_state=SEED).reset_index(drop=True)
    parts = np.array_split(pool.index, 4)
    # W2: เศรษฐกิจถดถอย คนคะแนนเครดิต/รายได้ต่ำมาสมัครมากขึ้น → สุ่มแบบถ่วงน้ำหนัก ทำให้ P(X) เปลี่ยนแต่ P(y|X) เท่าเดิม
    w = np.exp(-(pool["Credit_Score"] - 300) / 120) * np.exp(-pool["Annual_Income"] / 60000)
    week2 = pool.sample(len(parts[1]), replace=True, weights=w, random_state=SEED)
    # สุ่มแบบใส่คืนทำให้แถวซ้ำ → ให้เลขคำขอใหม่ต่อจากเลขล่าสุด (ถือเป็นผู้สมัครคนใหม่ที่หน้าตาคล้ายเดิม)
    start = int(df["Application_ID"].max()) + 1
    week2["Application_ID"] = np.arange(start, start + len(week2))
    batches = {
        "week_1_normal.csv": pool.loc[parts[0]],
        "week_2_data_drift.csv": week2,
        "week_3_concept_drift.csv": concept_drift_policy(pool.loc[parts[2]]),
        "week_4_concept_drift.csv": concept_drift_policy(pool.loc[parts[3]]),
        "bad_data.csv": make_bad_batch(pool),
    }
    for name, b in batches.items():
        b.to_csv(PRODUCTION / name, index=False, lineterminator="\n")
    return {"historical": len(hist), "pool": len(pool), "batches": {k: len(v) for k, v in batches.items()}}
