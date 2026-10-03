# เช็กลิสต์วันสาธิต (12 ต.ค. 2569)

## ก่อนเข้าห้อง (เช้าวันนำเสนอ หรือคืนก่อน)

1. `git switch main && git pull` บนเครื่องที่ใช้สาธิต (ห้ามอยู่บน branch อื่น — โฟลเดอร์ถูก mount เข้า container)
2. เปิด Docker Desktop แล้วล้างระบบให้สะอาด
   ```bash
   uv run python scripts/reset_demo.py --yes --build
   ```
3. ซ้อมรอบเต็มหนึ่งครั้ง (ประมาณ 6 นาที) ให้ระบบมี champion และ Grafana มีข้อมูล
   ```bash
   uv run python scripts/demo.py
   ```
4. ถ้าอยากเริ่มสาธิตจากศูนย์ต่อหน้าอาจารย์ ให้ reset อีกรอบก่อนเข้าห้อง แล้วสาธิตด้วย `scripts/demo.py`
5. ถ้าใช้ Discord: ใส่ `ALERT_WEBHOOK_URL` ใน `.env` (ดู `.env.example`) แล้วเปิดช่อง Discord ไว้บนจอ

## แท็บที่เปิดรอไว้

| แท็บ | URL | ใช้ตอน |
|---|---|---|
| Airflow | http://localhost:8080 | ดู task แดงตอนข้อมูลเสีย, กด rollback |
| MLflow | http://localhost:5050 | ดูเวอร์ชันและ alias |
| Grafana | http://localhost:3000 | drift / AUC / latency |
| Prometheus | http://localhost:9090/alerts | alert firing |
| สไลด์ | (ลิงก์สไลด์ของกลุ่ม) | |

## ระหว่างสาธิต

| ขั้น | คำสั่ง / ปุ่ม |
|---|---|
| ทั้งหมดตามลำดับ | `uv run python scripts/demo.py` |
| เฉพาะข้อมูลเสีย + concept drift (เวลาน้อย) | `uv run python scripts/demo.py --only 2 5` |
| Rollback | Airflow → `loan_rollback_pipeline` → Trigger (เว้น `to_version` ว่าง = กลับไป @previous) |
| Test case ของอาจารย์ | `uv run python scripts/run_test_cases.py --file <ไฟล์>.csv` (ดู `docs/test_cases.md`) |
| ให้ alert latency firing | `uv run python scripts/load_test.py --stress --n 3000 --concurrency 40` |

## แผนสำรอง

- Docker ช้า/ค้าง: ใช้ภาพใน `docs/evidence/` เล่าแทน
- API ไม่มีโมเดล (`/health` = 503): Trigger `loan_training_pipeline` แล้วรอ ~1 นาที
- Airflow ไม่ขึ้น DAG: `docker compose restart dag-processor scheduler`
