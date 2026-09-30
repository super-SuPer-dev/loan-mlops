"""การแจ้งเตือน — เขียนลงไฟล์ include/alerts.jsonl ทุกครั้ง และส่ง webhook ถ้าตั้ง ALERT_WEBHOOK_URL ไว้

ตั้ง ALERT_WEBHOOK_URL เป็น URL ของ Discord/Slack webhook ในไฟล์ .env ได้ ถ้าไม่ตั้งก็ยังมีบันทึกในไฟล์
"""
import json
import os
from datetime import datetime, timezone

import requests

from src.config import ALERTS_LOG


def send_alert(level: str, title: str, details=None) -> dict:
    alert = {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "level": level,          # INFO / WARNING / CRITICAL
        "title": title,
        "details": details or {},
    }
    ALERTS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(ALERTS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(alert, ensure_ascii=False) + "\n")
    print(f"🔔 ALERT [{level}] {title}: {json.dumps(details, ensure_ascii=False)}")

    url = os.getenv("ALERT_WEBHOOK_URL")
    if url:
        try:
            # รูปแบบ {"content": ...} ใช้ได้กับ Discord webhook
            requests.post(url, json={"content": f"[{level}] {title}\n{json.dumps(details, ensure_ascii=False)[:1500]}"},
                          timeout=5)
        except requests.RequestException as e:   # ส่งไม่ได้ไม่ควรทำให้ pipeline ล้มซ้ำ
            print("ส่ง webhook ไม่สำเร็จ:", e)
    return alert


def airflow_failure_callback(context) -> None:
    """ใส่ไว้ใน default_args ของ DAG: task ไหนล้ม (เช่น ข้อมูลเสีย) จะแจ้งเตือนอัตโนมัติ"""
    ti = context["task_instance"]
    send_alert("CRITICAL", f"Airflow task ล้ม: {ti.dag_id}.{ti.task_id}",
               {"run_id": context["run_id"], "error": str(context.get("exception"))[:1000]})
