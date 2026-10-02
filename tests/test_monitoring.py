"""ทดสอบหัวใจของ monitoring: แยก data drift / concept drift / ข้อมูลเสีย ให้ถูก

เตรียมระบบจำลองในโฟลเดอร์ชั่วคราว: ข้อมูลอดีต + schema + champion ใน MLflow (SQLite)
แล้วส่ง batch 4 แบบเข้า monitoring.check_batch
"""
import numpy as np
import pytest
from conftest import make_loans

from src import config as C
from src import monitoring, steps
from src.data_prep import make_bad_batch
from src.features import TARGET
from src.validation import DataValidationError


@pytest.fixture()
def system(tmp_path, monkeypatch):
    """ระบบจำลองที่มี champion แล้ว — คืนฟังก์ชันสำหรับวาง batch ลงโฟลเดอร์ data/"""
    data = tmp_path / "data"
    data.mkdir()
    make_loans(2000, seed=0).to_csv(data / "historical.csv", index=False)
    monkeypatch.setattr(C, "DATA", data)
    monkeypatch.setattr(C, "HISTORICAL_CSV", data / "historical.csv")
    monkeypatch.setattr(C, "RUNS", tmp_path / "include" / "runs")
    monkeypatch.setattr(C, "SCHEMA_PATH", tmp_path / "include" / "schema.json")
    monkeypatch.setattr(C, "MONITORING_LATEST", tmp_path / "include" / "monitoring" / "latest.json")
    monkeypatch.setattr(C, "MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.setattr(C, "MIN_ROC_AUC", 0.80)            # ข้อมูลสังเคราะห์ ใช้เกณฑ์ที่เหมาะกับมัน
    monkeypatch.setattr(C, "MIN_RECALL_REJECTED", 0.70)
    monkeypatch.setattr(C, "MAX_P95_LATENCY_MS", 1000.0)   # เทสต์นี้ไม่ได้ทดสอบ latency (เครื่อง CI ช้าได้)
    monkeypatch.setattr(steps, "CANDIDATES", {"logreg": steps.CANDIDATES["logreg"]})

    ex = steps.example_gen("seed")
    schema = steps.schema_gen(ex)
    best = steps.select_best([steps.trainer("logreg", ex, schema, "seed")])
    metrics = steps.evaluator(ex, best)
    assert metrics["blessed"], metrics["checks"]
    steps.register(best, metrics, promote=True)

    def put(df, name):
        df.to_csv(data / name, index=False)
        return name
    return put


def test_no_champion_does_not_retrain(tmp_path, monkeypatch):
    """ระบบเพิ่งตั้ง ยังไม่มีโมเดล → ห้าม retrain ด้วย batch เล็ก ๆ แค่แจ้งเตือน"""
    monkeypatch.setattr(C, "MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'empty.db').as_posix()}")
    result = monitoring.check_batch("anything.csv")
    assert result["needs_retrain"] is False and result["alerts"]


def test_normal_batch_has_no_alert(system):
    result = monitoring.check_batch(system(make_loans(1500, seed=7), "normal.csv"))
    assert not result["data_drift"] and not result["concept_drift"]
    assert result["alerts"] == [] and result["needs_retrain"] is False


def test_data_drift_alerts_but_does_not_retrain(system):
    """ข้อมูลเข้าเปลี่ยน (P(X)) ในฟีเจอร์ที่ไม่ได้กำหนด label → แจ้งเตือน แต่ AUC ยังดี จึงไม่ retrain"""
    df = make_loans(1500, seed=7)
    for col, factor in [("Annual_Income", 0.6), ("Savings_Balance", 0.4), ("Checking_Balance", 0.4),
                        ("Total_Assets", 0.5), ("Loan_Amount", 0.5)]:
        df[col] = (df[col] * factor).round(2)
    df["Monthly_Income"] = (df["Annual_Income"] / 12).round(2)     # รักษากฎความสอดคล้องของ schema
    df["Age"] = (df["Age"] * 0.8).round().clip(lower=18).astype(int)
    df["Region"] = "North"
    result = monitoring.check_batch(system(df, "drift.csv"))
    assert result["data_drift"] is True, result["drifted_features"]
    assert result["concept_drift"] is False and result["needs_retrain"] is False


def test_concept_drift_triggers_retrain(system):
    """ข้อมูลเข้าเหมือนเดิม แต่นโยบายใหม่เปลี่ยน label (P(y|X)) → AUC ตก → ต้อง retrain"""
    df = make_loans(1500, seed=7)
    df.loc[df["Loan_Term_Months"] > 60, TARGET] = "Rejected"
    result = monitoring.check_batch(system(df, "concept.csv"))
    assert result["concept_drift"] is True and result["needs_retrain"] is True
    assert result["drift_share"] <= C.DRIFT_SHARE_LIMIT             # ข้อมูลเข้าไม่ได้เปลี่ยน


def test_bad_batch_stops_monitoring(system):
    with pytest.raises(DataValidationError):
        monitoring.check_batch(system(make_bad_batch(make_loans(600, seed=7)), "bad.csv"))


def test_report_is_written_for_grafana(system):
    result = monitoring.check_batch(system(make_loans(800, seed=3), "w.csv"))
    run = C.run_dir("test-run")                                    # DAG จริงก็ใช้ run_dir ซึ่งสร้างโฟลเดอร์ให้
    path = monitoring.write_report(result, run)
    assert (run / "monitoring.json").exists() and path.endswith("monitoring.json")
    assert C.MONITORING_LATEST.exists()                            # API อ่านไฟล์นี้ส่งให้ Prometheus
    assert np.isfinite(result["roc_auc"])
