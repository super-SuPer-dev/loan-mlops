"""Rollback DAG — ย้อน @champion กลับเวอร์ชันก่อนหน้าด้วยปุ่มเดียวบนหน้าเว็บ Airflow

rollback_champion ─► reload_api ─► notify

ต่างจาก scripts/rollback.py ตรงที่ทุกครั้งที่ย้อนจะมีบันทึกใน Airflow (ใคร/เมื่อไร/ผลเป็นอย่างไร)
และกดได้จากหน้าเว็บโดยไม่ต้องเปิด terminal
"""
from __future__ import annotations

import os

import pendulum
import requests
from airflow.sdk import Param, dag, task

from src import steps
from src.alerts import airflow_failure_callback, send_alert

API_URL = os.getenv("API_URL", "http://api:8000")

DEFAULT_ARGS = {
    "owner": "mlops-loan-team",
    "retries": 1,
    "retry_delay": pendulum.duration(seconds=10),
    "on_failure_callback": airflow_failure_callback,
}


@dag(
    dag_id="loan_rollback_pipeline",
    description="ย้อนโมเดล champion กลับไปเวอร์ชันก่อนหน้า แล้วสั่ง API โหลดใหม่",
    schedule=None,                         # สั่งเองเท่านั้น ไม่มีรอบอัตโนมัติ
    start_date=pendulum.datetime(2026, 9, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,                     # กันกดซ้อนกันแล้ว alias สลับไปมา
    default_args=DEFAULT_ARGS,
    tags=["mlops", "rollback", "loan"],
    params={
        "to_version": Param("", type="string",
                            description="เวอร์ชันที่จะย้อนไป เช่น 2 · เว้นว่าง = กลับไป @previous"),
    },
)
def loan_rollback_pipeline():
    # retries=0: ถ้าลองซ้ำ การ rollback รอบสองจะสลับกลับไปเวอร์ชันเดิม (ไม่ idempotent)
    @task(task_id="rollback_champion", retries=0)
    def t_rollback(**context) -> dict:
        to_version = context["params"]["to_version"].strip() or None
        result = steps.rollback(to_version)
        print(f"ย้าย @champion: v{result['from']} → v{result['to']}")
        return result

    @task(task_id="reload_api")
    def t_reload(result: dict) -> dict:
        """ต้องสำเร็จ ไม่งั้น API ยังใช้โมเดลเดิมอยู่ → ให้ task ล้มและ retry"""
        r = requests.post(f"{API_URL}/reload", timeout=60)
        r.raise_for_status()
        info = r.json()
        if str(info.get("version")) != str(result["to"]):
            raise RuntimeError(f"API โหลดเวอร์ชัน {info.get('version')} แต่ควรเป็น {result['to']}")
        print("API โหลดแล้ว:", info)
        return info

    @task(task_id="notify")
    def t_notify(result: dict, reload_info: dict, **context) -> None:
        send_alert("WARNING", f"ROLLBACK champion v{result['from']} → v{result['to']}",
                   {"run_id": context["run_id"], "api_loaded_at": reload_info.get("loaded_at")})

    result = t_rollback()
    t_notify(result, t_reload(result))


loan_rollback_pipeline()
