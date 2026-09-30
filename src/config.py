"""ค่าคงที่ เส้นทางไฟล์ และเกณฑ์ทุกตัวของระบบ — ประกาศไว้ที่เดียว (หลักเดียวกับ common.py ใน Lab 10)

DAG, Flask API, สคริปต์ และเทสต์ อ่านค่าจากไฟล์นี้ทั้งหมด ถ้าจะเปลี่ยนเกณฑ์ให้แก้ที่นี่ที่เดียว
"""
import os
from pathlib import Path

# ใน container ของ Airflow โปรเจกต์ถูก mount ไว้ที่ /opt/project, ถ้ารันบนเครื่องตัวเองใช้โฟลเดอร์แม่ของ src/
ROOT = Path(os.getenv("ML_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
DATA = ROOT / "data"
RAW_CSV = DATA / "raw" / "loan_approval_dataset.csv"   # ไฟล์จาก Kaggle (ไม่ commit ลง git)
HISTORICAL_CSV = DATA / "historical.csv"             # 80% ข้อมูลอดีต ใช้เทรน
POOL_CSV = DATA / "production_pool.csv"              # 20% กันไว้จำลองข้อมูล production
PRODUCTION = DATA / "production"                     # batch รายสัปดาห์ที่จำลองแล้ว (มีเฉลยตามมาทีหลัง)
INCLUDE = ROOT / "include"                           # artifact ที่ pipeline สร้าง
RUNS = INCLUDE / "runs"                              # แยกโฟลเดอร์ตาม run_id ของ Airflow → ย้อนดูได้ทุกรอบ
SCHEMA_PATH = INCLUDE / "schema.json"                # schema กลาง สร้างครั้งเดียวแล้วใช้ตลอด
ALERTS_LOG = INCLUDE / "alerts.jsonl"                # ประวัติการแจ้งเตือนทั้งหมด
MONITORING_LATEST = INCLUDE / "monitoring" / "latest.json"   # ผล monitoring ล่าสุด → API ส่งต่อให้ Prometheus

KAGGLE_URL = ("https://www.kaggle.com/api/v1/datasets/download/"
              "mmumairkhattak/loan-approval-dataset-2026-credit-risk-and-bank-a")

# ---- MLflow (Lab 9) ----
# ใน docker compose ชี้ไปที่ MLflow server, บนเครื่องตัวเองใช้ไฟล์ SQLite
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
EXPERIMENT = "loan-approval"
MODEL_NAME = "loan-approval"
CHAMPION = "champion"
# MLflow 3 บันทึกโมเดลแบบ skops ต้องประกาศคลาสที่เราไว้ใจ (คลาสของเราเอง + โครงสร้างต้นไม้ที่เราเทรนเอง)
TRUSTED_TYPES = [
    "src.features.LoanFeatureEngineer", "numpy.dtype", "sklearn.tree._tree.Tree",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
]

SEED = 42

# ---- SLO ของ API (วัดด้วย scripts/load_test.py) ----
SLO_P95_MS = 200.0           # p95 ของ /predict ต้องไม่เกิน 200 ms ...
SLO_TARGET_RPS = 20          # ... ที่โหลด 20 คำขอ/วินาที (ประมาณการคำขอสินเชื่อช่วงพีคของธนาคารขนาดกลาง)
SLO_ERROR_RATE = 0.01

# ---- ด่านตรวจก่อนขึ้นใช้งาน (Gating metrics) — ค่ามาจากการทดลองใน notebook ----
MIN_ROC_AUC = 0.85
MIN_RECALL_REJECTED = 0.80
MIN_IMPROVEMENT = -0.003     # แย่กว่า champion ได้ไม่เกินเท่านี้ (วัดบน test set ชุดเดียวกัน)
# เวลาทำนาย 1 คำขอในโปรเซส: ผูกกับ SLO → ที่โหลดเป้าหมาย API ต้องไม่ยุ่งเกิน 50% ของเวลา
# (1000 ms / (2 × 20 คำขอ/วินาที) = 25 ms) ไม่งั้นคำขอต่อคิวแล้ว p95 พุ่ง (เจอจริงตอนทดสอบ: HGB ~35 ms → p95 195 ms)
MAX_P95_LATENCY_MS = 1000 / (2 * SLO_TARGET_RPS)
MAX_MODEL_SIZE_MB = 50.0
MAX_APPROVAL_GAP = 0.10      # fairness: อัตราอนุมัติระหว่างเพศต่างกันได้ไม่เกินนี้

# ---- ต้นทุนความผิดพลาด ใช้เลือก threshold ----
COST_FALSE_APPROVE = 3
COST_FALSE_REJECT = 1
GRAY_ZONE = 0.10             # ± รอบ threshold = ส่งเจ้าหน้าที่พิจารณา (cascade)

# ---- Monitoring ----
DRIFT_PVALUE = 0.01
DRIFT_SHARE_LIMIT = 0.20
AUC_DROP_LIMIT = 0.05


def run_dir(run_id: str) -> Path:
    """โฟลเดอร์ artifact ของ 1 run (ตั้งชื่อตาม run_id ของ Airflow แบบ Lab 11)"""
    d = RUNS / run_id.replace(":", "-").replace("+", "_")
    d.mkdir(parents=True, exist_ok=True)
    return d
