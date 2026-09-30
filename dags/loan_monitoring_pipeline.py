"""Monitoring DAG — ตรวจ batch ล่าสุด ถ้าเจอ concept drift ให้สั่ง training DAG (ปิดวงจร MLOps แบบ Lab 11)

check_batch ─► send_alerts ─► needs_retrain (short circuit) ─► trigger_training
"""
from __future__ import annotations

import json

import pendulum
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sdk import Param, dag, task
from airflow.sdk.exceptions import AirflowFailException

from src import monitoring
from src.alerts import airflow_failure_callback, send_alert
from src.config import run_dir
from src.validation import DataValidationError

DEFAULT_ARGS = {
    "owner": "mlops-loan-team",
    "retries": 1,
    "retry_delay": pendulum.duration(seconds=20),
    "on_failure_callback": airflow_failure_callback,
}


@dag(
    dag_id="loan_monitoring_pipeline",
    description="ตรวจ data drift / concept drift ของ batch ล่าสุด และสั่ง retrain เมื่อถึงเกณฑ์",
    schedule="0 1 * * *",                  # ทุกวัน ตี 1 (UTC) — ถี่กว่าการเทรน
    start_date=pendulum.datetime(2026, 9, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["mlops", "monitoring", "loan"],
    params={
        "batch_file": Param("production/week_1_normal.csv", type="string",
                            description="batch ที่มีเฉลยแล้ว: week_1_normal / week_2_data_drift / "
                                        "week_3_concept_drift / week_4_concept_drift / bad_data"),
    },
)
def loan_monitoring_pipeline():
    @task(task_id="check_batch", retries=0)
    def t_check_batch(**context) -> dict:
        try:
            result = monitoring.check_batch(context["params"]["batch_file"])
        except DataValidationError as e:
            raise AirflowFailException(str(e))   # ข้อมูลเสีย → หยุด ไม่ retrain
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("รายงาน:", monitoring.write_report(result, run_dir(context["run_id"])))
        return result

    @task(task_id="send_alerts")
    def t_send_alerts(result: dict) -> dict:
        for msg in result.get("alerts", []):
            send_alert("CRITICAL" if "CONCEPT" in msg else "WARNING", msg,
                       {"batch": result["batch_file"], "drift_share": result.get("drift_share"),
                        "roc_auc": result.get("roc_auc")})
        return result

    @task.short_circuit(task_id="needs_retrain")
    def t_needs_retrain(result: dict) -> bool:
        print("ต้อง retrain" if result["needs_retrain"] else "ยังอยู่ในเกณฑ์ ไม่ต้อง retrain")
        return bool(result["needs_retrain"])

    trigger_training = TriggerDagRunOperator(
        task_id="trigger_training",
        trigger_dag_id="loan_training_pipeline",
        conf={"data_file": "{{ params.batch_file }}"},   # retrain ด้วยข้อมูลล่าสุดที่มีเฉลย (sliding window)
        wait_for_completion=False,
    )

    result = t_send_alerts(t_check_batch())
    t_needs_retrain(result) >> trigger_training


loan_monitoring_pipeline()
