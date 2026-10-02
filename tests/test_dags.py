"""ทดสอบ DAG (แบบ Lab 11) — ต้องมี Airflow จึงรันใน container:
    docker compose exec scheduler pytest /opt/project/tests/test_dags.py -v
บนเครื่องที่ไม่มี Airflow เทสต์นี้จะถูกข้ามอัตโนมัติ
"""
import pytest

pytest.importorskip("airflow")

from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {"loan_training_pipeline", "loan_monitoring_pipeline", "loan_rollback_pipeline"}
# task ที่ตั้งใจไม่ retry: ข้อมูลเสียลองกี่ทีก็เสีย / rollback ซ้ำ = สลับกลับไปเวอร์ชันเดิม
NO_RETRY = {"example_validator", "check_batch", "rollback_champion"}


@pytest.fixture(scope="session")
def dagbag():
    from airflow.models import DagBag
    # known_pools=set() ทำให้ไม่ต้องต่อฐานข้อมูล metadata ตอนทดสอบ (แบบ Lab 11)
    return DagBag(dag_folder=str(ROOT / "dags"), known_pools=set())


def test_no_import_errors(dagbag):
    assert not dagbag.import_errors, dagbag.import_errors


def test_expected_dags_exist(dagbag):
    assert EXPECTED.issubset(dagbag.dag_ids)


def test_owner_and_retries(dagbag):
    for dag_id in EXPECTED:
        for t in dagbag.dags[dag_id].tasks:
            assert t.owner != "airflow"
            if t.task_id not in NO_RETRY:
                assert t.retries >= 1, t.task_id


def test_validation_before_training(dagbag):
    dag = dagbag.dags["loan_training_pipeline"]
    assert "trainer" in dag.get_task("example_validator").downstream_task_ids
    assert dag.get_task("blessing_gate").downstream_task_ids == {"promote", "register_rejected"}


def test_monitoring_triggers_training(dagbag):
    t = dagbag.dags["loan_monitoring_pipeline"].get_task("trigger_training")
    assert t.trigger_dag_id in dagbag.dag_ids


# ------------------------------------------------------------------ Rollback DAG
def test_rollback_is_manual_only(dagbag):
    dag = dagbag.dags["loan_rollback_pipeline"]
    assert dag.schedule is None or str(dag.schedule) == "None"   # สั่งเองเท่านั้น
    assert dag.max_active_runs == 1                              # กันกดซ้อนแล้ว alias สลับไปมา


def test_rollback_task_order(dagbag):
    dag = dagbag.dags["loan_rollback_pipeline"]
    assert dag.get_task("rollback_champion").downstream_task_ids == {"reload_api", "notify"}
    assert dag.get_task("reload_api").downstream_task_ids == {"notify"}


def test_rollback_is_not_retried(dagbag):
    dag = dagbag.dags["loan_rollback_pipeline"]
    assert dag.get_task("rollback_champion").retries == 0
    assert dag.get_task("reload_api").retries >= 1
