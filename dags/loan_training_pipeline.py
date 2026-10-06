"""Training DAG — จากข้อมูลดิบจนถึงโมเดลที่ให้บริการ กดปุ่มเดียว (ต่อยอด adult_training_pipeline ของ Lab 11)

example_gen → schema_gen → example_validator ─(ข้อมูลเสีย: ล้ม + แจ้งเตือน)
                                  │
                                  ▼
          trainer[logreg] / trainer[logreg_c0.1] / trainer[hist_gboost]   (รันขนาน, dynamic task mapping)
                                  │
                              select_best
                           ┌──────┴──────┐
                       evaluator   fairness_audit
                           │
                     blessing_gate ──► promote (@champion) ──┐
                           └────────► register_rejected ─────┴─► reload_api ─► report
"""
from __future__ import annotations

import json
import os

import pendulum
import requests
from airflow.sdk import Param, dag, task
from airflow.sdk.exceptions import AirflowFailException
from airflow.task.trigger_rule import TriggerRule

from src import steps
from src.alerts import airflow_failure_callback, send_alert

API_URL = os.getenv("API_URL", "http://api:8000")

DEFAULT_ARGS = {
    "owner": "mlops-loan-team",
    "retries": 1,
    "retry_delay": pendulum.duration(seconds=20),
    "on_failure_callback": airflow_failure_callback,   # task ไหนล้ม → แจ้งเตือนอัตโนมัติ
}


@dag(
    dag_id="loan_training_pipeline",
    description="เทรน ประเมิน และอนุมัติโมเดลทำนายการอนุมัติสินเชื่อ",
    schedule="0 2 * * 1",                  # retrain ตามรอบทุกวันจันทร์ ตี 2 (UTC) แม้ไม่มี alert
    start_date=pendulum.datetime(2026, 9, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,                     # กันสอง run แย่งกันเปลี่ยน @champion
    default_args=DEFAULT_ARGS,
    tags=["mlops", "training", "loan"],
    params={
        "data_file": Param("historical.csv", type="string",
                           description="ไฟล์ในโฟลเดอร์ data/ เช่น production/bad_data.csv เพื่อสาธิตด่านตรวจข้อมูล"),
    },
)
def loan_training_pipeline():
    @task(task_id="example_gen")
    def t_example_gen(**context) -> dict:
        return steps.example_gen(context["run_id"], context["params"]["data_file"])

    @task(task_id="schema_gen")
    def t_schema_gen(examples: dict) -> str:
        return steps.schema_gen(examples)

    @task(task_id="example_validator", retries=0)
    def t_example_validator(examples: dict, schema_path: str) -> dict:
        result = steps.example_validator(examples, schema_path)
        for a in result["anomalies"]:
            print("ANOMALY:", a)
        if not result["ok"]:
            # AirflowFailException = ล้มทันทีไม่ retry เพราะข้อมูลเสียลองกี่ครั้งก็เสีย
            # task ที่ตามมาทั้งหมด (เทรน, deploy) จะไม่ถูกรัน และ on_failure_callback จะแจ้งเตือน
            raise AirflowFailException("ข้อมูลไม่ผ่าน schema: " + " | ".join(result["anomalies"]))
        return result

    @task(task_id="trainer")
    def t_trainer(candidate: str, examples: dict, schema_path: str, **context) -> dict:
        return steps.trainer(candidate, examples, schema_path, context["run_id"])

    @task(task_id="select_best")
    def t_select_best(results, examples: dict) -> dict:
        return steps.select_best(list(results), examples)

    @task(task_id="evaluator")
    def t_evaluator(examples: dict, best: dict) -> dict:
        m = steps.evaluator(examples, best)
        print(json.dumps(m, indent=2, default=str))
        return m

    @task(task_id="fairness_audit")
    def t_fairness_audit(examples: dict, best: dict) -> dict:
        result = steps.fairness_audit(examples, best)
        print(result)
        if not result["ok"]:
            send_alert("WARNING", "Fairness เกินเกณฑ์", result)
        return result

    @task.branch(task_id="blessing_gate")
    def t_blessing_gate(metrics: dict) -> str:
        print("ผลด่านตรวจ:", metrics["checks"])
        return "promote" if metrics["blessed"] else "register_rejected"

    @task(task_id="promote")
    def t_promote(best: dict, metrics: dict, **context) -> dict:
        info = steps.register(best, metrics, promote=True, airflow_run_id=context["run_id"])
        send_alert("INFO", f"โมเดลเวอร์ชัน {info['version']} ขึ้นเป็น champion", info)
        return info

    @task(task_id="register_rejected")
    def t_register_rejected(best: dict, metrics: dict, **context) -> dict:
        info = steps.register(best, metrics, promote=False, airflow_run_id=context["run_id"])
        send_alert("WARNING", f"โมเดลเวอร์ชัน {info['version']} ไม่ผ่านด่านตรวจ champion เดิมให้บริการต่อ",
                   {"checks": metrics["checks"]})
        return info

    @task(task_id="reload_api", trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)
    def t_reload_api() -> None:
        """บอก Flask API ให้โหลด @champion ใหม่ — API ล่มก็ไม่ถือว่า pipeline ล้ม เพราะตอนเริ่มมันโหลดเองอยู่แล้ว"""
        try:
            r = requests.post(f"{API_URL}/reload", timeout=60)
            print("API reload:", r.status_code, r.text)
        except requests.RequestException as e:
            print("ติดต่อ API ไม่ได้ (ข้ามได้):", e)

    @task(task_id="report", trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)
    def t_report(best: dict, metrics: dict, fairness: dict, **context) -> None:
        status = "PROMOTED" if metrics["blessed"] else "REJECTED"
        print(f"run={context['run_id']} | {status} | model={best['candidate']} | test_auc={metrics['roc_auc']:.4f} "
              f"| champion_auc={metrics['champion_roc_auc']} | fairness_gap={fairness['gap']}")

    # ------------------------------------------------------------ ผูกลำดับงาน
    examples = t_example_gen()
    schema = t_schema_gen(examples)
    validation = t_example_validator(examples, schema)
    trained = t_trainer.partial(examples=examples, schema_path=schema).expand(candidate=list(steps.CANDIDATES))
    validation >> trained                      # ต้องผ่านด่านข้อมูลก่อนจึงเทรนได้
    best = t_select_best(trained, examples)
    metrics = t_evaluator(examples, best)
    fairness = t_fairness_audit(examples, best)
    gate = t_blessing_gate(metrics)
    promote = t_promote(best, metrics)
    rejected = t_register_rejected(best, metrics)
    reload = t_reload_api()
    gate >> [promote, rejected]
    [promote, rejected] >> reload >> t_report(best, metrics, fairness)


loan_training_pipeline()
