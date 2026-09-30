"""Flask API ให้บริการทำนายการอนุมัติสินเชื่อ (แนวเดียวกับ Lab 3 / Lab 7, endpoint แบบ Lab 10)

GET  /health          ตรวจสุขภาพ: โหลดโมเดลแล้วหรือยัง เวอร์ชันไหน
GET  /model           ข้อมูลโมเดลที่ให้บริการ (เวอร์ชัน, threshold, test AUC)
POST /predict         ทำนาย 1 คำขอ   → cascade: AUTO_APPROVE / AUTO_REJECT / MANUAL_REVIEW
POST /predict/batch   ทำนายหลายคำขอ (งานกลางคืน)
POST /reload          โหลด @champion ใหม่จาก MLflow (Airflow เรียกหลัง promote / หลัง rollback)
GET  /metrics         ตัวเลขให้ Prometheus เก็บ (จำนวนคำขอ, error, latency, การตัดสินใจ)

กัน Training-Serving Skew: โมเดลที่โหลดคือ sklearn Pipeline ก้อนเดียวที่มีการแปลงข้อมูลอยู่ข้างใน
และ schema ที่ใช้ตรวจคำขอก็ดาวน์โหลดจาก run เดียวกับที่เทรนโมเดล
"""
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import pandas as pd
from flask import Flask, Response, jsonify, request
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_client.core import GaugeMetricFamily

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # ให้ import src ได้ทั้งใน container และบนเครื่อง
from src import config as C  # noqa: E402
from src import validation as V  # noqa: E402

# ---- Metrics สำหรับ Prometheus ----
REQUESTS = Counter("loan_api_requests_total", "จำนวนคำขอ", ["endpoint", "status"])
LATENCY = Histogram("loan_api_latency_seconds", "เวลาตอบกลับ", ["endpoint"],
                    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1, 2))
DECISIONS = Counter("loan_api_decisions_total", "ผลการตัดสินใจ", ["decision"])
PROBA = Histogram("loan_api_probability", "ความน่าจะเป็นที่ทำนาย (ใช้ดู prediction drift)",
                  buckets=[i / 10 for i in range(11)])
MODEL_VERSION = Gauge("loan_api_model_version", "เวอร์ชันโมเดลที่ให้บริการ")
MODEL_LOADED = Gauge("loan_api_model_loaded", "1 = มีโมเดลพร้อมให้บริการ")
# metric เฉพาะโจทย์ธนาคาร (แบบงาน 6.1 ของ Lab 10): ผลตัดสินแยกตามสถานะการจ้างงาน
# ใช้ดูว่ากลุ่มไหนถูกปฏิเสธผิดปกติ เช่น ถ้า Self-Employed ถูก AUTO_REJECT พุ่งขึ้นกะทันหัน
DECISIONS_BY_SEGMENT = Counter("loan_api_decisions_by_employment_total", "ผลตัดสินแยกตามสถานะการจ้างงาน",
                               ["employment_status", "decision"])

# เกณฑ์จาก src/config.py ส่งออกเป็น metric → กฎของ Prometheus อ้างอิงค่าชุดเดียวกับโค้ด (ไม่ต้องพิมพ์เลขซ้ำ)
THRESHOLDS = Gauge("loan_config_threshold", "เกณฑ์ของระบบจาก src/config.py", ["name"])
for _name, _value in {
    "slo_p95_seconds": C.SLO_P95_MS / 1000, "slo_error_rate": C.SLO_ERROR_RATE,
    "drift_share_limit": C.DRIFT_SHARE_LIMIT, "auc_drop_limit": C.AUC_DROP_LIMIT,
    "min_roc_auc": C.MIN_ROC_AUC,
}.items():
    THRESHOLDS.labels(_name).set(_value)


class MonitoringCollector:
    """อ่านผล monitoring DAG ล่าสุด (include/monitoring/latest.json) ทุกครั้งที่ Prometheus มาเก็บ metric

    ทำให้ Grafana แสดงได้ทั้งสถานะระบบ (จาก API) และคุณภาพโมเดล (จาก monitoring DAG) ในหน้าเดียว
    """

    FIELDS = {"drift_share": "สัดส่วนฟีเจอร์ที่ drift", "roc_auc": "AUC ของ champion บน batch ล่าสุด",
              "reference_roc_auc": "AUC ตอน deploy (เส้นฐาน)", "auc_drop": "AUC ที่ตกลงจากเส้นฐาน",
              "data_drift": "1 = พบ data drift", "concept_drift": "1 = พบ concept drift",
              "checked_at": "เวลาที่ตรวจล่าสุด (unix)"}

    def collect(self):
        try:
            data = json.loads(C.MONITORING_LATEST.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for field, doc in self.FIELDS.items():
            if data.get(field) is not None:
                g = GaugeMetricFamily(f"loan_monitor_{field}", doc, labels=["batch"])
                g.add_metric([str(data.get("batch_file", ""))], float(data[field]))
                yield g


REGISTRY.register(MonitoringCollector())

# ---- Log แบบ JSON ทีละบรรทัด: stdout (docker logs) + ไฟล์ (ใช้ทำ monitoring ภายหลัง) ----
LOG_DIR = Path(os.getenv("PREDICTION_LOG_DIR", C.ROOT / "logs" / "api"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
log = logging.getLogger("loan_api")
log.setLevel(logging.INFO)
for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_DIR / "predictions.jsonl", encoding="utf-8")):
    h.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(h)

app = Flask(__name__)
STATE = {"model": None, "schema": None, "version": None, "threshold": 0.5, "test_roc_auc": None, "loaded_at": None}
LOCK = threading.Lock()


def load_champion() -> dict:
    """โหลดโมเดล @champion + threshold + schema จาก MLflow registry แล้วสลับเข้า STATE ทีเดียว"""
    mlflow.set_tracking_uri(C.MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()
    mv = client.get_model_version_by_alias(C.MODEL_NAME, C.CHAMPION)
    model = mlflow.sklearn.load_model(f"models:/{C.MODEL_NAME}@{C.CHAMPION}")
    schema_file = mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path="schema.json")
    new_state = {"model": model, "schema": V.load_schema(schema_file), "version": mv.version,
                 "threshold": float(mv.tags.get("threshold", 0.5)), "test_roc_auc": mv.tags.get("test_roc_auc"),
                 "loaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with LOCK:                                   # สลับทีเดียว คำขอที่กำลังทำงานไม่เห็นสถานะครึ่ง ๆ กลาง ๆ
        STATE.update(new_state)
    MODEL_VERSION.set(int(mv.version))
    MODEL_LOADED.set(1)
    log.info(json.dumps({"event": "model_loaded", "version": mv.version, "threshold": STATE["threshold"]}))
    return {k: v for k, v in STATE.items() if k not in ("model", "schema")}


def load_with_retry(attempts: int = 30, wait_s: float = 10) -> None:
    """ตอนเริ่มระบบครั้งแรก MLflow อาจยังไม่พร้อม หรือยังไม่มี champion → ลองใหม่เรื่อย ๆ ใน background"""
    for _ in range(attempts):
        try:
            load_champion()
            return
        except Exception as e:  # noqa: BLE001 — ยังไม่มีโมเดลไม่ใช่เหตุให้ API ล่ม
            print("ยังโหลด champion ไม่ได้:", str(e)[:200], flush=True)
            time.sleep(wait_s)


def decide(p: float, threshold: float) -> str:
    """Cascade: มั่นใจ → ตัดสินอัตโนมัติ, ก้ำกึ่ง → ส่งเจ้าหน้าที่"""
    if p >= threshold + C.GRAY_ZONE:
        return "AUTO_APPROVE"
    if p < threshold - C.GRAY_ZONE:
        return "AUTO_REJECT"
    return "MANUAL_REVIEW"


def predict_frame(df: pd.DataFrame) -> list[dict]:
    with LOCK:
        model, threshold, ver = STATE["model"], STATE["threshold"], STATE["version"]
    probs = model.predict_proba(df)[:, 1]
    segments = df["Employment_Status"].fillna("Unknown").astype(str).tolist()
    out = []
    for p, seg in zip(probs, segments):
        d = decide(float(p), threshold)
        DECISIONS.labels(d).inc()
        DECISIONS_BY_SEGMENT.labels(seg, d).inc()
        PROBA.observe(float(p))
        out.append({"probability_approve": round(float(p), 4), "decision": d,
                    "approved": bool(p >= threshold), "model_version": ver})
    return out


def handle(endpoint: str, fn):
    """ครอบทุก endpoint: จับเวลา, นับ metric, เขียน log, แปลง exception เป็น JSON"""
    t0 = time.perf_counter()
    try:
        body, status = fn()
    except Exception as e:  # noqa: BLE001
        body, status = {"error": f"internal error: {e}"}, 500
    elapsed = time.perf_counter() - t0
    REQUESTS.labels(endpoint, str(status)).inc()
    LATENCY.labels(endpoint).observe(elapsed)
    if endpoint.startswith("/predict"):
        log.info(json.dumps({"time": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                             "endpoint": endpoint, "status": status, "latency_ms": round(elapsed * 1000, 2),
                             "request": request.get_json(silent=True), "response": body}, ensure_ascii=False, default=str))
    return jsonify(body), status


def model_ready():
    return STATE["model"] is not None


@app.get("/health")
def health():
    ok = model_ready()
    return jsonify({"status": "ok" if ok else "loading", "model_loaded": ok, "model_version": STATE["version"],
                    "loaded_at": STATE["loaded_at"]}), (200 if ok else 503)


@app.get("/model")
def model_info():
    return jsonify({k: v for k, v in STATE.items() if k not in ("model", "schema")})


@app.post("/predict")
def predict():
    def run():
        if not model_ready():
            return {"error": "ยังไม่มีโมเดลให้บริการ"}, 503
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return {"error": "ต้องส่ง JSON object ของคำขอ 1 รายการ"}, 400
        df = pd.DataFrame([payload])
        problems = V.validate(df, STATE["schema"], batch=False)
        if problems:
            return {"error": "ข้อมูลไม่ผ่าน schema", "details": problems}, 400
        return predict_frame(df)[0], 200
    return handle("/predict", run)


@app.post("/predict/batch")
def predict_batch():
    def run():
        if not model_ready():
            return {"error": "ยังไม่มีโมเดลให้บริการ"}, 503
        payload = request.get_json(silent=True)
        if not isinstance(payload, list) or not payload:
            return {"error": "ต้องส่ง JSON array ของคำขอ"}, 400
        df = pd.DataFrame(payload)
        problems = V.validate(df, STATE["schema"], batch=False)
        if problems:
            return {"error": "ข้อมูลไม่ผ่าน schema", "details": problems}, 400
        return {"n": len(df), "predictions": predict_frame(df)}, 200
    return handle("/predict/batch", run)


@app.post("/reload")
def reload_model():
    try:
        return jsonify({"reloaded": True, **load_champion()}), 200
    except Exception as e:  # noqa: BLE001
        return jsonify({"reloaded": False, "error": str(e)}), 500


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    threading.Thread(target=load_with_retry, daemon=True).start()
    # threaded=True ให้ Flask รับหลายคำขอพร้อมกันได้ (เพียงพอสำหรับโครงงาน)
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), threaded=True)
