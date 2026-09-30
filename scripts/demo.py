"""รันการสาธิตทั้งระบบตามลำดับที่ถูกต้อง (ใช้ซ้อมและใช้วันนำเสนอ)

    docker compose up -d --build                 # ยกระบบก่อน
    uv run python scripts/demo.py                # รันทุกขั้น
    uv run python scripts/demo.py --only 3 4     # รันเฉพาะบางขั้น

ขั้นตอน
  1 เทรนรอบแรก (ข้อมูลอดีต)            → โมเดลแรกขึ้นเป็น champion
  2 ข้อมูลเสีย                          → pipeline หยุดที่ example_validator + ALERT
  3 W1 ปกติ                             → ไม่มี alert
  4 W2 data drift + ส่ง traffic กลุ่มใหม่ → ALERT data drift (ไม่ retrain), Grafana เห็นสัดส่วนปฏิเสธพุ่ง
  5 W3 concept drift                    → ALERT concept drift → training DAG ถูกสั่งเอง → champion ใหม่
  6 W4 หลัง retrain                     → AUC กลับมาดี
  7 load test ตาม SLO                   → p50 / p95 / throughput
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
API = "http://localhost:8000"


def airflow(*args) -> str:
    cmd = ["docker", "compose", "exec", "-T", "scheduler", "airflow", *args]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8").stdout


def runs(dag: str) -> list[dict]:
    out = airflow("dags", "list-runs", dag, "-o", "json")
    try:
        return json.loads(out[out.index("["):])
    except ValueError:
        return []


def trigger(dag: str, run_id: str, conf: dict | None = None) -> None:
    args = ["dags", "trigger", dag, "--run-id", run_id]
    if conf:
        args += ["-c", json.dumps(conf)]
    airflow(*args)
    print(f"  ▶ trigger {dag} ({run_id}) {conf or ''}")


def wait_run(dag: str, run_id: str, timeout: int = 900) -> str:
    t0 = time.time()
    while time.time() - t0 < timeout:
        state = next((r["state"] for r in runs(dag) if r["run_id"] == run_id), "queued")
        if state in ("success", "failed"):
            print(f"  ■ {run_id}: {state}")
            return state
        time.sleep(5)
    raise TimeoutError(run_id)


def wait_training_idle(timeout: int = 900) -> None:
    """รอจน training DAG ไม่มี run ค้าง (ใช้หลัง monitoring สั่ง retrain เอง)"""
    t0 = time.time()
    time.sleep(10)
    while time.time() - t0 < timeout:
        active = [r for r in runs("loan_training_pipeline") if r["state"] in ("queued", "running")]
        if not active:
            latest = runs("loan_training_pipeline")[0]
            print(f"  ■ retrain {latest['run_id']}: {latest['state']}")
            return
        time.sleep(5)


def monitoring_result(run_id: str) -> dict:
    return json.loads((ROOT / "include" / "runs" / run_id / "monitoring.json").read_text(encoding="utf-8"))


def show_monitoring(run_id: str) -> None:
    r = monitoring_result(run_id)
    print(f"  drift_share={r.get('drift_share')}  auc={r.get('roc_auc')}  (deploy {r.get('reference_roc_auc')})  "
          f"retrain={r.get('needs_retrain')}  alerts={r.get('alerts')}")


def model() -> dict:
    return requests.get(f"{API}/model", timeout=10).json()


def load_test(*extra: str) -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts" / "load_test.py"), *extra], cwd=ROOT)


def step(n: int, title: str) -> None:
    print(f"\n=== {n}. {title} " + "=" * max(0, 60 - len(title)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, nargs="*", help="เลขขั้นที่จะรัน (ไม่ใส่ = ทุกขั้น)")
    args = ap.parse_args()
    todo = set(args.only or range(1, 8))
    stamp = time.strftime("%H%M%S")

    for dag in ("loan_training_pipeline", "loan_monitoring_pipeline"):
        airflow("dags", "unpause", dag)

    if 1 in todo:
        step(1, "เทรนรอบแรกจากข้อมูลอดีต")
        trigger("loan_training_pipeline", f"demo_{stamp}_initial")
        wait_run("loan_training_pipeline", f"demo_{stamp}_initial")
        time.sleep(3)
        print("  API:", model())
    if 2 in todo:
        step(2, "ข้อมูลเสีย → pipeline ต้องหยุด")
        trigger("loan_training_pipeline", f"demo_{stamp}_bad", {"data_file": "production/bad_data.csv"})
        wait_run("loan_training_pipeline", f"demo_{stamp}_bad")
        last = (ROOT / "include" / "alerts.jsonl").read_text(encoding="utf-8").strip().splitlines()[-1]
        print("  ALERT ล่าสุด:", json.loads(last)["title"])
    for n, week, title in [(3, "week_1_normal", "W1 ปกติ"), (4, "week_2_data_drift", "W2 data drift"),
                           (5, "week_3_concept_drift", "W3 concept drift"),
                           (6, "week_4_concept_drift", "W4 หลัง retrain")]:
        if n not in todo:
            continue
        step(n, title)
        if n == 4:   # ส่ง traffic จากผู้สมัครกลุ่มใหม่เข้า API 60 วินาที ให้ Grafana เห็น prediction drift
            print("  ส่ง traffic กลุ่ม W2 เข้า API 60 วินาที ...")
            load_test("--source", "data/production/week_2_data_drift.csv", "--duration", "60", "--rps", "10")
        run_id = f"demo_{stamp}_{week}"
        trigger("loan_monitoring_pipeline", run_id, {"batch_file": f"production/{week}.csv"})
        wait_run("loan_monitoring_pipeline", run_id)
        show_monitoring(run_id)
        if n == 5:
            wait_training_idle()
            time.sleep(3)
            print("  API หลัง retrain:", model())
    if 7 in todo:
        step(7, "Load test ตาม SLO (20 คำขอ/วินาที, 30 วินาที)")
        load_test()
    print("\nดูผล: Airflow http://localhost:8080 | MLflow http://localhost:5050 | "
          "Grafana http://localhost:3000 | Prometheus alerts http://localhost:9090/alerts")


if __name__ == "__main__":
    main()
