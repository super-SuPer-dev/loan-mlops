# หลักฐานสำหรับรายงานและสไลด์

| ไฟล์ | แสดงอะไร | ใช้กับหัวข้อ |
|---|---|---|
| `01_ci_pass_main.png` | CI บน `main` ผ่านครบ 5 job (push แรก, 4m 29s) | CI/CD: ครั้งที่ผ่าน |
| `02_ci_fail_pr_checks.png` | PR #1: model-quality ไม่ผ่าน, deliver ถูกข้าม, อีก 3 job ผ่าน | CI/CD: ครั้งที่ไม่ผ่าน |
| `03_ci_fail_pipeline.png` | กราฟ job ของ run ที่ไม่ผ่าน | CI/CD: ครั้งที่ไม่ผ่าน |
| `04_ci_fail_model_gate.png` | ตาราง Model quality gate: ROC-AUC 0.8994 < 0.95 → ไม่ผ่าน, exit code 1 | CI/CD, gating metric |
| `05_airflow_bad_data_stops.jpg` | Airflow grid: run ที่ใช้ `bad_data.csv` แดงที่ `example_validator` และ task ถัดไปไม่ถูกรัน | Data validation |
| `06_mlflow_registry_aliases.jpg` | MLflow registry: v3 `@champion`, v2 `@previous` พร้อม tag status/threshold/test AUC | Model registry, rollback |
| `07_prometheus_alert_firing.jpg` | Prometheus: `LoanLatencyP95AboveSLO` FIRING ระหว่าง stress test 40 ผู้ใช้พร้อมกัน | Monitoring ด้านระบบ, SLO |
| `08_grafana_system_health.jpg` | Grafana: สถานะระบบ (model version, req/s, p95, 5xx) | Serving metrics |
| `09_grafana_predictions.jpg` | Grafana: ผลตัดสิน cascade, สัดส่วนปฏิเสธแยกตามอาชีพ, heatmap ความน่าจะเป็น | Prediction monitoring |
| `10_grafana_model_quality.jpg` | Grafana: drift share พุ่งตอน W2, AUC ตกตอน W3 แล้วกลับหลัง retrain | Data drift vs concept drift |

ภาพทั้งหมดถ่ายจากระบบที่รันจริงเมื่อ 30 ก.ย. – 1 ต.ค. 2569
