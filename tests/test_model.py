"""ด่านคุณภาพโมเดล: รัน training pipeline ทั้งเส้น (ไม่ผ่าน Airflow) กับ MLflow ชั่วคราว
แล้วตรวจว่า gate, registry, rollback ทำงานถูก — CI ใช้ไฟล์นี้เป็นด่านเกณฑ์คุณภาพของโมเดล
"""
import pytest

from src import config as C
from src import steps


@pytest.fixture()
def project(tmp_path, monkeypatch, loans):
    """ชี้ทุก path และ MLflow ไปที่โฟลเดอร์ชั่วคราว ไม่ยุ่งกับของจริง"""
    data = tmp_path / "data"
    data.mkdir()
    loans.to_csv(data / "historical.csv", index=False)
    monkeypatch.setattr(C, "DATA", data)
    monkeypatch.setattr(C, "RUNS", tmp_path / "include" / "runs")
    monkeypatch.setattr(C, "SCHEMA_PATH", tmp_path / "include" / "schema.json")
    monkeypatch.setattr(C, "MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.setattr(C, "MIN_ROC_AUC", 0.80)          # ข้อมูลสังเคราะห์เล็ก ใช้เกณฑ์ที่เหมาะกับมัน
    monkeypatch.setattr(C, "MIN_RECALL_REJECTED", 0.70)
    monkeypatch.setattr(steps, "CANDIDATES", {"logreg": steps.CANDIDATES["logreg"]})
    return tmp_path


def run_pipeline(run_id: str) -> tuple[dict, dict]:
    ex = steps.example_gen(run_id)
    schema = steps.schema_gen(ex)
    assert steps.example_validator(ex, schema)["ok"]
    best = steps.select_best([steps.trainer("logreg", ex, schema, run_id)])
    return best, steps.evaluator(ex, best)


def test_split_is_reproducible(project):
    a, b = steps.example_gen("r1"), steps.example_gen("r2")
    assert open(a["train"]).read() == open(b["train"]).read()


def test_first_model_passes_gate_and_becomes_champion(project):
    best, m = run_pipeline("run1")
    assert m["blessed"], m["checks"]
    info = steps.register(best, m, promote=True)
    model, mv = steps.load_champion()
    assert mv.version == info["version"] and model is not None


def test_rollback_restores_previous_champion(project):
    best, m = run_pipeline("run1")
    v1 = steps.register(best, m, promote=True)["version"]
    best2, m2 = run_pipeline("run2")
    assert m2["champion_version"] == v1                      # รอบสองต้องเทียบกับ champion เดิม
    v2 = steps.register(best2, m2, promote=True)["version"]
    assert steps.load_champion()[1].version == v2
    assert steps.rollback() == {"from": v2, "to": v1}
    assert steps.load_champion()[1].version == v1


def test_gate_rejects_bad_model(project, monkeypatch):
    monkeypatch.setattr(C, "MIN_ROC_AUC", 0.999)             # เกณฑ์ที่ไม่มีทางผ่าน
    best, m = run_pipeline("run1")
    assert not m["blessed"] and not m["checks"]["roc_auc"]
