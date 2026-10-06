# Loan Approval Prediction: MLOps Pipeline

โครงงานรายวิชา CP413008 Machine Learning Engineering for Production

ระบบคัดกรองคำขอสินเชื่อ ทำนายว่าคำขอจะ **อนุมัติ (Approved)** หรือ **ปฏิเสธ (Rejected)**
ชุดข้อมูล: [Loan Approval Dataset 2026 (Kaggle, MIT)](https://www.kaggle.com/datasets/mmumairkhattak/loan-approval-dataset-2026-credit-risk-and-bank-a) มี 30,000 แถว

## สถาปัตยกรรม

```
                ┌──────────────────── Airflow (http://localhost:8080) ────────────────────┐
 Kaggle CSV ──► │ loan_training_pipeline                                                 │
 data/          │  example_gen → schema_gen → example_validator ─(ข้อมูลเสีย)→ ล้ม + ALERT │
                │   → trainer ×3 (ขนาน) → select_best → evaluator / fairness_audit        │
                │   → blessing_gate → promote | register_rejected → reload_api → report   │
                │                                                                         │
                │ loan_monitoring_pipeline                                                │
                │  check_batch (schema + KS/Chi² drift + AUC) → send_alerts               │
                │   → needs_retrain → TriggerDagRun(loan_training_pipeline)              │
                └──────┬───────────────────────────────┬───────────────────┬──────────────┘
                       │ log runs / register / alias    │ POST /reload      │ include/monitoring/latest.json
                       ▼                                ▼                   ▼
        MLflow (http://localhost:5050) ─@champion─► Flask API (http://localhost:8000)
        tracking + model registry                   /health /predict /predict/batch /reload /metrics
                                                              │ scrape ทุก 5 วินาที
                                                              ▼
                                   Prometheus (http://localhost:9090) ──► Grafana (http://localhost:3000)
                                   กฎแจ้งเตือน 7 ข้อ                          dashboard ระบบ + โมเดล

 GitHub Actions: ruff + pytest → ตรวจข้อมูลจริง → ด่านคุณภาพโมเดล / ทดสอบ DAG → build + push image
```

| หน้าที่ | เครื่องมือ | เคยใช้ใน |
|---|---|---|
| Version control | Git / GitHub (branch + Pull Request) | Lab 9 |
| Containerization | Docker, docker compose | Lab 3, 8, 11 |
| Data validation | schema.json + `src/validation.py` | Lab 11 |
| Experiment tracking + Registry | MLflow (alias champion / previous / challenger) | Lab 9 |
| Orchestration | Airflow 3 (TaskFlow, branch, dynamic mapping, TriggerDagRun) | Lab 11 |
| Serving | Flask ใน Docker + cascade | Lab 3, 7 |
| Monitoring | KS / Chi-square drift + Prometheus + Grafana | Lab 10, 11 |
| CI/CD | GitHub Actions + ruff + pytest | Lab 9 |

## รันจากเครื่องเปล่า

ต้องมี: Docker Desktop และ [uv](https://docs.astral.sh/uv/) (ใช้รันเทสต์และสคริปต์บนเครื่อง)

```bash
git clone <repo-url> && cd loan_mlops
uv sync
docker compose up -d --build        # ครั้งแรกใช้เวลาหลายนาที (build image + ดาวน์โหลดข้อมูลจาก Kaggle)
```

| บริการ | URL |
|---|---|
| Airflow | http://localhost:8080 |
| MLflow | http://localhost:5050 (บน Windows พอร์ต 5000 ถูกระบบจองไว้) |
| API | http://localhost:8000/health |
| Prometheus (หน้า Alerts) | http://localhost:9090/alerts |
| Grafana | http://localhost:3000 |

เปิด Airflow แล้วเปิดใช้ (unpause) และกด **Trigger** ที่ `loan_training_pipeline`
**กดปุ่มเดียวนี้จะได้ครบทุกขั้น ตั้งแต่ข้อมูลดิบ ตรวจสอบ เทรน ประเมิน อนุมัติ จนถึงให้บริการ**

ทดลองเรียก API:
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @scripts/sample_request.json
```

## สาธิตทั้งระบบด้วยคำสั่งเดียว

```bash
uv run python scripts/demo.py              # รันทุกขั้นตามลำดับ (~6 นาที)
uv run python scripts/demo.py --only 4 5   # รันเฉพาะบางขั้น
```

| # | สาธิต | ผลที่ควรเห็น |
|---|---|---|
| 1 | เทรนรอบแรก | โมเดลแรกขึ้นเป็น champion, API โหลดเอง |
| 2 | ข้อมูลเสีย (`production/bad_data.csv`) | `example_validator` สีแดง, pipeline หยุด, ALERT ใน `include/alerts.jsonl` |
| 3 | W1 ปกติ | drift ~0%, ไม่มี alert |
| 4 | W2 data drift + ส่ง traffic กลุ่มใหม่เข้า API | ALERT data drift แต่ **ไม่** retrain (AUC ยังดี), Grafana เห็นสัดส่วนปฏิเสธพุ่ง |
| 5 | W3 concept drift | AUC ตก → ALERT → training DAG ถูกสั่งเอง → champion ใหม่ → API reload |
| 6 | W4 หลัง retrain | AUC กลับมาสูง ไม่มี alert |
| 7 | Load test ตาม SLO | p50 / p95 / throughput ที่ 20 คำขอ/วินาที → `reports/load_test.json` |

**ต้องรันตามลำดับ** เพราะหลัง W3 โมเดลถูก retrain ตามนโยบายใหม่ ถ้ารัน W2 ทีหลังจะเห็น concept drift ปนมาด้วย

สาธิตเพิ่มเติม:

| สาธิต | คำสั่ง |
|---|---|
| Rollback | `uv run python scripts/rollback.py` (champion กลับไป `@previous`, API โหลดใหม่ทันที) |
| ความจุสูงสุด | `uv run python scripts/load_test.py --stress` |
| ทำให้กฎ latency ของ Prometheus firing | `uv run python scripts/load_test.py --stress --n 3000 --concurrency 40` |
| เทสต์ (ข้อมูล / โมเดล / API) | `uv run pytest -q` |
| เทสต์ DAG | `docker compose exec scheduler python -m pytest /opt/project/tests/test_dags.py -v` |
| ล้างทุกอย่างเริ่มใหม่ | `docker compose down -v` |

## SLO

| ตัวชี้วัด | เป้าหมาย | วัดด้วย |
|---|---|---|
| Latency p95 ของ `/predict` | ≤ 200 ms ที่โหลด 20 คำขอ/วินาที | `scripts/load_test.py`, Grafana, กฎ `LoanLatencyP95AboveSLO` |
| Error rate (5xx) | ≤ 1% | `scripts/load_test.py`, กฎ `LoanErrorRateAboveSLO` |
| Availability | `/health` ตอบ 200 | Docker healthcheck, กฎ `LoanApiDown` |

ด่าน latency ของโมเดลผูกกับ SLO: `MAX_P95_LATENCY_MS = SLO_P95_MS / 2 = 100 ms`
วัดเป็น p95 **ที่โหลดเป้าหมาย 20 คำขอ/วินาที** (คำขอเข้าพร้อมกันหลาย thread แบบที่ API เจอ) อีกครึ่งของ SLO เผื่อไว้ให้ HTTP และการตรวจ schema
`select_best` วัดทีละโมเดลแล้วตัดตัวที่เกินงบนี้ทิ้งก่อนเลือกตาม AUC

ทำไมไม่วัดทีละคำขอ (เจอจริง): HistGradientBoosting วัดทีละคำขอได้ ~17 ms เร็วพอ ๆ กับ LogReg (~12 ms)
แต่ที่ 20 คำขอ/วินาที p95 ในโปรเซสขึ้นเป็น ~150 ms (LogReg ~15 ms) และ p95 ของ API เป็น 222 ms หลุด SLO
ด่านแบบทีละคำขอจึงปล่อยโมเดลนี้ผ่านมาได้ — เกณฑ์ที่ใช้ตัดสินต้องวัดในสภาพเดียวกับที่ให้บริการจริง

เกณฑ์ทุกตัว (gate, drift, SLO) อยู่ใน [src/config.py](src/config.py) ที่เดียว
API ส่งเกณฑ์เหล่านี้ออกเป็น metric `loan_config_threshold` ทำให้กฎของ Prometheus อ้างอิงค่าชุดเดียวกัน

## Monitoring (Prometheus + Grafana)

| กฎแจ้งเตือน ([monitoring/alert_rules.yml](monitoring/alert_rules.yml)) | ด้าน | เงื่อนไข |
|---|---|---|
| `LoanApiDown` | ระบบ | API ไม่ตอบ 30 วินาที |
| `LoanModelNotLoaded` | ระบบ | API ยังไม่มีโมเดล 1 นาที |
| `LoanLatencyP95AboveSLO` | ระบบ | p95 ของ `/predict` > 200 ms |
| `LoanErrorRateAboveSLO` | ระบบ | 5xx > 1% |
| `LoanDataDrift` | โมเดล | ฟีเจอร์ drift > 20% (จาก monitoring DAG) |
| `LoanConceptDrift` | โมเดล | AUC ตก > 0.05 (จาก monitoring DAG) |
| `LoanRejectRateSpike` | โมเดล | สัดส่วน AUTO_REJECT > 70% ใน 5 นาที (prediction drift ไม่ต้องรอเฉลย) |

Dashboard (สร้างจาก [monitoring/grafana/build_dashboard.py](monitoring/grafana/build_dashboard.py)) มี 3 ส่วน:
สถานะระบบ · การตัดสินใจของโมเดล (รวม metric เฉพาะโจทย์: ผลตัดสินแยกตามสถานะการจ้างงาน) · คุณภาพโมเดลจาก monitoring DAG

## CI/CD (GitHub Actions)

[.github/workflows/ci.yml](.github/workflows/ci.yml) รันทุก push และ Pull Request

| Job | ตรวจอะไร |
|---|---|
| `code-quality` | ruff + pytest ทั้งหมด (ข้อมูลสังเคราะห์) |
| `data-validation` | ดาวน์โหลดข้อมูลจริง ตรวจกับ schema และข้อมูลเสียต้องถูกจับได้ (`scripts/ci_checks.py data`) |
| `model-quality` | เทรนบนข้อมูลจริง ต้องผ่านด่านตรวจทุกข้อ (`scripts/ci_checks.py model`) |
| `dag-tests` | build image ของ Airflow แล้วทดสอบ DAG ข้างใน |
| `deliver` | build image ของ API + smoke test และ push ขึ้น ghcr.io เมื่อ merge เข้า main |

ผลของ data-validation และ model-quality แสดงเป็นตารางในหน้าสรุปของแต่ละ run

**หลักฐานครั้งที่ไม่ผ่าน** (โจทย์ต้องการ) เปิด PR ที่ตั้งใจทำให้พัง แล้วเก็บภาพหน้าจอ เช่น
- แก้ `MIN_ROC_AUC = 0.95` ใน `src/config.py` → `model-quality` แดง
- ลบกฎ `Credit_Score` ออกจาก `DOMAIN_RULES` ใน `src/validation.py` → `code-quality` แดง (pytest จับได้)
- เพิ่ม import ที่ไม่ได้ใช้ → ruff แดง

## โครงสร้างโปรเจกต์

```
src/          โค้ดที่ใช้ร่วมกันทั้ง DAG และ API (features, validation, steps, monitoring, config, alerts)
dags/         Airflow DAG 2 ตัว (มีแค่ลำดับงาน ตรรกะอยู่ใน src/)
serving/      Flask API + Dockerfile
monitoring/   Prometheus config + กฎแจ้งเตือน, Grafana datasource + dashboard
scripts/      demo, load test, rollback, CI checks
tests/        pytest: ข้อมูล, โมเดล/registry, API, DAG
notebooks/    ต้นแบบ pipeline ทั้งเส้น พร้อมคำอธิบาย
docker/       Dockerfile ของ Airflow และ MLflow
.github/      GitHub Actions workflow
```
