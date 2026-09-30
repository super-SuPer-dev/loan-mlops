"""ย้อน @champion กลับไปเวอร์ชันก่อนหน้า แล้วสั่ง API โหลดใหม่ทันที

    uv run python scripts/rollback.py                 # กลับไปที่ @previous
    uv run python scripts/rollback.py --to-version 1  # กลับไปเวอร์ชันที่ระบุ
"""
import argparse
import os
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5050")   # MLflow server ใน docker compose
from src import steps  # noqa: E402
from src.alerts import send_alert  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--to-version")
    ap.add_argument("--api", default="http://localhost:8000")
    args = ap.parse_args()
    result = steps.rollback(args.to_version)
    send_alert("WARNING", f"ROLLBACK champion v{result['from']} → v{result['to']}", result)
    r = requests.post(f"{args.api}/reload", timeout=60)
    print("API reload:", r.status_code, r.json())


if __name__ == "__main__":
    main()
