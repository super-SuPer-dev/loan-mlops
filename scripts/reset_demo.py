"""ล้างระบบให้สะอาดก่อนซ้อม/ก่อนนำเสนอ แล้วยกขึ้นใหม่ รอจนทุกบริการพร้อม

    uv run python scripts/reset_demo.py            # ถามยืนยันก่อนลบ
    uv run python scripts/reset_demo.py --yes      # ไม่ถาม
    uv run python scripts/reset_demo.py --yes --build   # build image ใหม่ด้วย (หลังแก้โค้ด API/Dockerfile)

สิ่งที่ถูกลบ: registry + ประวัติการทดลองใน MLflow, ฐานข้อมูล Airflow, Prometheus/Grafana,
include/ (schema, artifact ของแต่ละ run, alerts), logs/, reports/
สิ่งที่ไม่ถูกลบ: data/raw/ (ไฟล์จาก Kaggle) — ไม่ต้องดาวน์โหลดใหม่
ต้องรันจากโฟลเดอร์โปรเจกต์ที่มี docker-compose.yml
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ["include", "logs/airflow", "logs/api", "reports"]
READY = {                                   # บริการ → URL ที่ต้องตอบก่อนถือว่าพร้อม
    "MLflow": "http://localhost:5050/health",
    "Airflow": "http://localhost:8080/api/v2/version",
    "Flask API": "http://localhost:8000/health",   # ยังไม่มีโมเดลจะได้ 503 = โปรเซสพร้อมแล้ว
    "Prometheus": "http://localhost:9090/-/ready",
    "Grafana": "http://localhost:3000/api/health",
}


def compose(*args: str) -> None:
    print("$ docker compose", " ".join(args))
    subprocess.run(["docker", "compose", *args], cwd=ROOT, check=True)


def wait_ready(timeout_s: int = 600) -> bool:
    pending, t0 = dict(READY), time.time()
    while pending and time.time() - t0 < timeout_s:
        for name, url in list(pending.items()):
            try:
                if requests.get(url, timeout=3).status_code < 500 or name == "Flask API":
                    print(f"  ✓ {name} พร้อม ({int(time.time() - t0)} วินาที)")
                    pending.pop(name)
            except requests.RequestException:
                pass
        time.sleep(3)
    for name in pending:
        print(f"  ✗ {name} ยังไม่พร้อมหลัง {timeout_s} วินาที")
    return not pending


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="ไม่ต้องถามยืนยัน")
    ap.add_argument("--build", action="store_true", help="build image ใหม่ก่อนยกระบบ")
    args = ap.parse_args()
    if not (ROOT / "docker-compose.yml").exists():
        sys.exit("ไม่พบ docker-compose.yml ต้องรันในโฟลเดอร์โปรเจกต์")
    if not args.yes and input("ลบ registry, ประวัติ run และ log ทั้งหมด แล้วเริ่มใหม่? พิมพ์ yes: ").strip() != "yes":
        sys.exit("ยกเลิก")

    t0 = time.time()
    compose("down", "-v")                            # -v = ลบ volume ของ postgres, mlflow, prometheus, grafana
    for rel in CLEAN:
        shutil.rmtree(ROOT / rel, ignore_errors=True)
        (ROOT / rel).mkdir(parents=True, exist_ok=True)
    compose("up", "-d", *(["--build"] if args.build else []))

    print("รอทุกบริการพร้อม ...")
    ok = wait_ready()
    print(f"\n{'พร้อมแล้ว' if ok else 'บางบริการยังไม่พร้อม ดู docker compose ps'} ({int(time.time() - t0)} วินาที)")
    print("ขั้นต่อไป: uv run python scripts/demo.py")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
