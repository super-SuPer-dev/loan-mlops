"""วัด Latency p50/p95/p99 + Throughput + Error rate ของ API เทียบกับ SLO

    uv run python scripts/load_test.py                    # ยิงที่อัตราตาม SLO (20 คำขอ/วินาที) นาน 30 วินาที
    uv run python scripts/load_test.py --rps 40           # ลองโหลดสูงขึ้น
    uv run python scripts/load_test.py --stress           # ยิงเร็วที่สุด เพื่อหาความจุสูงสุด (throughput)

ปนคำขอเสีย 5% (แบบงานที่ 6 ของ Lab 10) เพื่อดูว่า API ตอบ 400 ได้ถูกและไม่ล่ม
คำขอเสียที่ได้ 400 ถือว่า "ตอบถูก" ส่วน error ที่นับใน SLO คือ 5xx และ timeout
"""
import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src import features as F  # noqa: E402


def make_payloads(n: int, bad_ratio: float, source: Path) -> list[dict]:
    """คำขอจริงจากไฟล์ source ส่งเฉพาะคอลัมน์ที่ลูกค้ากรอกตอนยื่น (ไม่มีคอลัมน์ที่รู้หลังตัดสินใจ)"""
    cols = F.NUMERIC + F.CATEGORICAL + ["Monthly_Income"]
    rows = pd.read_csv(source)[cols].sample(n, replace=True, random_state=C.SEED)
    payloads = json.loads(rows.to_json(orient="records"))
    for i in np.random.default_rng(C.SEED).choice(n, int(n * bad_ratio), replace=False):
        payloads[i]["Credit_Score"] = 999
    return payloads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--rps", type=float, default=C.SLO_TARGET_RPS, help="อัตราคำขอต่อวินาที (โหมดปกติ)")
    ap.add_argument("--duration", type=float, default=30, help="วินาที (โหมดปกติ)")
    ap.add_argument("--stress", action="store_true", help="ยิงต่อเนื่องไม่เว้นช่วง เพื่อหาความจุสูงสุด")
    ap.add_argument("--n", type=int, default=1000, help="จำนวนคำขอ (โหมด stress)")
    ap.add_argument("--concurrency", type=int, default=10, help="จำนวนผู้ใช้พร้อมกัน (โหมด stress)")
    ap.add_argument("--bad-ratio", type=float, default=0.05)
    ap.add_argument("--source", default=str(C.POOL_CSV),
                    help="ไฟล์ที่ใช้สุ่มคำขอ เช่น data/production/week_2_data_drift.csv เพื่อจำลองผู้สมัครกลุ่มที่เปลี่ยนไป")
    args = ap.parse_args()

    n = args.n if args.stress else int(args.rps * args.duration)
    payloads = make_payloads(n, args.bad_ratio, Path(args.source))
    local = threading.local()

    def call(p):
        if not hasattr(local, "s"):
            local.s = requests.Session()
        t0 = time.perf_counter()
        try:
            status = local.s.post(f"{args.url}/predict", json=p, timeout=10).status_code
        except requests.RequestException:
            status = 599
        return (time.perf_counter() - t0) * 1000, status

    t0 = time.perf_counter()
    if args.stress:
        with ThreadPoolExecutor(args.concurrency) as pool:
            results = list(pool.map(call, payloads))
    else:
        # ส่งคำขอตามนาฬิกา: คำขอที่ i ออกเวลา i/rps วินาที ไม่ว่าคำขอก่อนหน้าจะตอบแล้วหรือยัง (เหมือนลูกค้าจริง)
        futures = []
        with ThreadPoolExecutor(64) as pool:
            for i, p in enumerate(payloads):
                delay = t0 + i / args.rps - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                futures.append(pool.submit(call, p))
            results = [f.result() for f in futures]
    wall = time.perf_counter() - t0

    lat = np.array([r[0] for r in results])
    status = np.array([r[1] for r in results])
    report = {
        "mode": "stress" if args.stress else f"steady {args.rps:g} rps",
        "n_requests": n,
        "p50_ms": round(float(np.percentile(lat, 50)), 2), "p95_ms": round(float(np.percentile(lat, 95)), 2),
        "p99_ms": round(float(np.percentile(lat, 99)), 2), "throughput_rps": round(n / wall, 1),
        "status_counts": {str(k): int(v) for k, v in zip(*np.unique(status, return_counts=True))},
        "error_rate_5xx": round(float((status >= 500).mean()), 4),
        "slo": {"p95_ms": C.SLO_P95_MS, "at_rps": C.SLO_TARGET_RPS, "error_rate": C.SLO_ERROR_RATE},
    }
    report["slo_met"] = report["p95_ms"] <= C.SLO_P95_MS and report["error_rate_5xx"] <= C.SLO_ERROR_RATE
    out = C.ROOT / "reports" / ("load_test_stress.json" if args.stress else "load_test.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not args.stress:
        print("✅ ผ่าน SLO" if report["slo_met"] else "❌ ไม่ผ่าน SLO")


if __name__ == "__main__":
    main()
