# รายการแก้ไขสไลด์และรายงาน (7 ต.ค. 2569)

ไฟล์ในโฟลเดอร์นี้

| ไฟล์ | คืออะไร |
|---|---|
| `slides_original.pdf` | สไลด์ฉบับเดิม (export จาก Canva ล่าสุดที่มีในเครื่อง: `Loan_Approval_mlops_slide-1.pdf` เวลา 23:36 น. 6 ต.ค., 33 หน้า) |
| `../report.pdf` | รายงานฉบับเดิม (ไฟล์เดิมใน `docs/` ไม่ได้แตะ, 28 หน้า) |
| `slides_fixed.pdf` | สไลด์ฉบับแก้ |
| `report_fixed.pdf` | รายงานฉบับแก้ |

ไฟล์ฉบับแก้สร้างโดยแก้ทับบน PDF เดิมทีละจุด ต้นฉบับใน Canva และ Google Docs ยังไม่ได้แก้ ถ้าจะแก้ที่ต้นฉบับ ใช้ตารางด้านล่างไล่แก้ตามได้
ข้อความที่แก้ใน PDF ถูกวางเป็นภาพ จึงเลือกหรือค้นข้อความตรงจุดนั้นไม่ได้

## สไลด์ (27 จุด)

| หน้า PDF | เดิม | ใหม่ |
|---|---|---|
| 5 | p95 latency ต่อ 1 คําขอ | p95 latency ที่ 20 req/s |
| 5 | ≤ 25 ms | ≤ 100 ms |
| 5 | มาจาก SLO 200 ms ทีˠ 20 req/s | ครึ่งหนึ่งของ SLO 200 ms |
| 13 | Defaulf_Risk | Default_Risk |
| 13 | week1_normal | week_1_normal |
| 13 | week2_data_drift | week_2_data_drift |
| 13 | week3_concept_drift | week_3_concept_drift |
| 13 | week4_concept_drift | week_4_concept_drift |
| 17 | lower_q = 0.1, upper_q = 0.99 | lower_q = 0.01 , upper_q = 0.99 |
| 17 | handle__unknow = “ignore” | handle_unknown = “ignore” |
| 17 | c = 0.01, max_iter = 2000 | C = 0.1 , max_iter = 2000 |
| 17 | n_estimators = 300, min_sample_leaf,n_jobs = -1, random_state = 42 | n_estimators = 300 , min_samples_leaf = 5 , n_jobs = -1 , random_state = 42 |
| 17 | max_iter = 300, learning_rate = 0.5, random_state = 42 | max_iter = 300 , learning_rate = 0.05 , random_state = 42 |
| 18 | select_best: ตัดตัวทีˠ p95 > 25 ms ก่อน แล้วเลือก logreg เว้นแต่ตัวอืˠนชนะ AUC เกิน 0.003 (RF ไม่เข้ารอบ เพราะใหญ่และช้า) | select_best: ตัดตัวที่ p95 ที่ 20 req/s เกิน 100 ms ก่อน แล้วเลือก logreg เว้นแต่ตัวอื่นชนะ AUC เกิน 0.003 (RF ไม่เข้ารอบ เพราะใหญ่และช้า) |
| 23 | หลักฐานจริง: v3 hist_gboost = @champion · v2 logreg = @previous (rollback ด้วย scripts/rollback.py) | หลักฐานจริง: v3 logreg = @champion · v2 logreg = @previous (rollback: scripts/rollback.py หรือ Rollback DAG) |
| 23 | (ภาพเดิม xref 156) | ภาพใหม่ registry_crop.png |
| 24 | 23.7 ms | 20.7 ms |
| 24 | 66.7 ms | 54.6 ms |
| 24 | w = 0.10 แต่ไม่เกินครึˠงของระยะทีˠเหลือถึง 0/1 (เช่น t = 0.72→ w ฝ˞ˠ งอนุมัติ = 0.04) | w = 0.10 แต่ไม่เกินครึ่งของระยะที่เหลือถึง 0/1 (เช่น t = 0.92 → w ฝั่งอนุมัติ = 0.04 ส่วน t = 0.72 → w = 0.10 ทั้งสองฝั่ง) |
| 24 | บทเรียน: HistGradientBoosting ใช้ ~35 ms ต่อคําขอ ทํา p95 พุ่งถึง 195 ms → ผูกด่าน latency ≤ 25 ms กับ SLO และเลือกโมเดลโดยดูความเร็วด้วย | บทเรียน: HistGradientBoosting ผ่านด่านเดิมที่วัดทีละคำขอ (~17 ms ≤ 25 ms) แต่ที่ 20 req/s p95 ของ API ขึ้นเป็น 222 ms หลุด SLO → เปลี่ยนด่านเป็นวัด p95 ที่โหลด 20 req/s ≤ 100 ms (ครึ่งหนึ่งของ SLO) หลังแก้ ระบบเลือก logreg และ p95 ของ API ≈ 55 ms |
| 25 | 0.931 | 0.919 |
| 25 | ไม่มี alert (โมเดล v3) | ไม่มี alert (โมเดล v3 logreg) |
| 28 | ruff + pytest 14 เทสต์ | ruff + pytest 32 เทสต์ |
| 29 | (เพิ่มใหม่) | ภาพจาก run ก่อนเปลี่ยนด่าน latency จึงยังแสดงเกณฑ์เดิม (≤ 25 ms ต่อ 1 คำขอ) ปัจจุบันคือ p95 ที่ 20 req/s ≤ 100 ms |
| 30 | (เพิ่มใหม่) | ภาพจาก run ก่อนเปลี่ยนด่าน latency จึงยังแสดงเกณฑ์เดิม (≤ 25 ms ต่อ 1 คำขอ) ปัจจุบันคือ p95 ที่ 20 req/s ≤ 100 ms |
| 32 | 6. W4 → AUC กลับมา 0.93 | 6. W4 → AUC กลับมา 0.92 |
| 32 | Flask โปรเซสเดียว ≈ 75 req/s | Flask โปรเซสเดียว ≈ 39 req/s |

## รายงาน (38 จุด)

| หน้า PDF | เดิม | ใหม่ |
|---|---|---|
| 1 | (#11) | (#11), ด่าน latency ใต้โหลด (#16), คู่มือรันระบบ (#18) |
| 4 | ชุดทดสอบ ส่วน API มี p95 latency 66.7 ms ที่โหลด 20 คำขอ/วินาที ผ่าน SLO 200 ms | ชุดทดสอบ ส่วน API มี p95 latency 54.6 ms ที่โหลด 20 คำขอ/วินาที ผ่าน SLO 200 ms |
| 6 | p95 latency ต่อ 1 คำขอ | p95 latency ที่โหลด 20 คำขอ/วินาที |
| 6 | ≤ 25 ms | ≤ 100 ms |
| 6 | มาจาก SLO 200 ms ที่ 20 คำขอ/วินาที: 20 × 25 ms = 0.5 วินาทีต่อ / วินาที ให้ API ไม่ยุ่งเกิน 50% | ครึ่งหนึ่งของ SLO 200 ms อีกครึ่งเผื่อให้ HTTP และการตรวจ schema |
| 6 | สัดส่วนคำขอที่ตัดสินอัตโนมัติ (ภาระงานเจ้าหน้าที่ที่ลดลง): 78.5% โดยตัดสินถูก 80.3% ในส่วนนี้ | สัดส่วนคำขอที่ตัดสินอัตโนมัติ (ภาระงานเจ้าหน้าที่ที่ลดลง): 78.5% โดยตัดสินถูก 80.3% ในส่วนนี้ (ค่าจาก notebook, threshold 0.77) |
| 6 | ความแม่นแยกโซน: AUTO_APPROVE 93.2% / AUTO_REJECT 74.9% โซน AUTO_APPROVE ควรแม่นสูงที่สุด เพราะอนุมัติผิดมีต้นทุนสูงสุด | ความแม่นแยกโซน (pipeline v2): AUTO_APPROVE 93.2% / AUTO_REJECT 74.9% โซน AUTO_APPROVE ควรแม่นสูงที่สุด เพราะอนุมัติผิดแพงที่สุด |
| 6 | p95 ≤ 200 ms ที่ 20 คำขอ/วินาที (ช่วงพีคของธนาคารขนาดกลาง) ความจุสูงสุดที่วัดได้ประมาณ 75 คำขอ/วินาที | p95 ≤ 200 ms ที่ 20 คำขอ/วินาที (ช่วงพีคของธนาคารขนาดกลาง) ความจุสูงสุดที่วัดได้ประมาณ 39 คำขอ/วินาที |
| 14 | (ภาพเดิม xref 44) | ภาพใหม่ registry_crop_r.png (v3 logreg = @champion) |
| 15 | Flask API ใน Docker ผ่าน SLO: p95 = 66.7 ms ที่ 20 คำขอ/วินาที (เกณฑ์ 200 ms) และ 5xx = 0% | Flask API ใน Docker ผ่าน SLO: p95 = 54.6 ms ที่ 20 คำขอ/วินาที (เกณฑ์ 200 ms) และ 5xx = 0% |
| 16 | LoanLatencyP95AboveSLO (ช่วงเวลา | LoanLatencyP95AboveSLO (วัดเมื่อ |
| 16 | 6 ต.ค ) | 6 ต.ค. 2569) |
| 16 | 66.7 ms (ผ่าน) | 54.6 ms (ผ่าน) |
| 16 | 54.6 ms (ผ่าน) | ตอบ 200 (ผ่าน) |
| 17 | 23.7 ms | 20.7 ms |
| 17 | 66.7 ms | 54.6 ms |
| 17 | 131.7 ms | 261 ms |
| 17 | 186.3 ms | 314 ms |
| 17 | 75.1 คำขอ/วินาที | 39.0 คำขอ/วินาที |
| 17 | 1,328 ms | 1,281 ms |
| 17 | 35.0 คำขอ/วินาที | 36.3 คำขอ/วินาที |
| 17 | ที่ 40 ผู้ใช้ throughput ลดเหลือ 35 คำขอ/วินาที เพราะ ที่ 40 ผู้ใช้ throughput อยู่ที่ 36 คำขอ/วินาที ใกล้เคียงกับที่ 10 ผู้ใช้ (39 คำขอ/วินาที) / แสดงว่าระบบถึงเพดานความจุแล้ว ผู้ใช้ที่เพิ่มขึ้นจึงไม่ได้งานเพิ่ม แต่ต้องรอคิวนานขึ้น p50 จึงเพิ่มจาก 261 ms เป็น 1,133 ms สาเหตุที่น่าจะเป็นคือ / API เป็น Flask โปรเซสเดียว และ Python ประมวลผลได้ทีละ thread (GIL) เมื่อแต่ละคำขอใช้เวลาประมวลผลราว 26 ms ความจุสูงสุดจึงอยู่ที่ / ประมาณ 38 คำขอ/วินาที | ที่ 40 ผู้ใช้ throughput อยู่ที่ 36 คำขอ/วินาที ใกล้เคียงกับที่ 10 ผู้ใช้ (39 คำขอ/วินาที) แสดงว่าระบบถึงเพดานความจุแล้ว ผู้ใช้ที่เพิ่มขึ้นจึงไม่ได้งานเพิ่ม แต่ต้องรอคิวนานขึ้น p50 จึงเพิ่มจาก 261 ms เป็น 1,133 ms สาเหตุที่น่าจะเป็นคือ API เป็น Flask โปรเซสเดียว และ Python ประมวลผลได้ ทีละ thread (GIL) เมื่อแต่ละคำขอใช้เวลาประมวลผลราว 26 ms ความจุสูงสุดจึงอยู่ที่ประมาณ 38 คำขอ/วินาที |
| 17 | ค่า p95 ที่ Prometheus แจ้งเตือน (1.9 วินาที) ต่างจาก load_test (1,328 ms) เพราะ Prometheus ประมาณจาก histogram bucket ฝั่ง / เซิร์ฟเวอร์ และคำนวณในช่วงเวลาที่ต่างกัน | ค่า p95 ที่ Prometheus แจ้งเตือน (1.69 วินาที) ต่างจาก load_test (1,281 ms) เพราะ Prometheus ประมาณจาก histogram bucket ฝั่งเซิร์ฟเวอร์ และคำนวณในช่วงเวลาที่ต่างกัน |
| 17 | (เพิ่มใหม่) | 7.6 บทเรียน: ด่าน latency ต้องวัดในสภาพเดียวกับที่ให้บริการ |
| 17 | (เพิ่มใหม่) | หลังส่งรายงานฉบับแรก เราพบว่าด่าน latency เดิมที่วัด p95 ทีละคำขอ (≤ 25 ms) ปล่อย HistGradientBoosting ผ่าน (16.7 ms) จนขึ้นเป็น champion ตอน retrain แต่เมื่อคำขอเข้าพร้อมกันที่ 20 คำขอ/วินาที p95 ของ API เป็น 222 ms หลุด SLO เราจึงเปลี่ยนด่านเป็นวัด p95 ที่โหลดเป้าหมาย 20 คำขอ/วินาที ≤ 100 ms (ครึ่งหนึ่งของ SLO) ทั้งใน select_best และ evaluator (PR #16) หลังแก้ ระบบตัด HistGradientBoosting (147 ms) และเลือก LogReg (16.7 ms) ตอน retrain ส่วน p95 ของ API ที่เสิร์ฟ v3 วัดได้ 55–57 ms ตารางและตัวเลขในฉบับนี้ปรับตามระบบหลังแก้แล้ว |
| 18 | v3 hist_gboost | v3 logreg |
| 18 | 0.931 (0.913) | 0.919 (0.890) |
| 19 | p95 > 200 ms (ทดสอบให้ firing แล้วที่ 1.9 วินาที) | p95 > 200 ms (ทดสอบให้ firing แล้วที่ 1.69 วินาที) |
| 20 | (ภาพเดิม xref 64) | ภาพใหม่ grafana_quality_rows.png (AUC 0.919) |
| 20 | รูปที่ 9 Grafana กลุ่มคุณภาพโมเดลหลัง retrain (W4): AUC บน batch ล่าสุด 0.931 และ drift share 0% | รูปที่ 9 Grafana กลุ่มคุณภาพโมเดลหลัง retrain (W4): AUC บน batch ล่าสุด 0.919 และ drift share 0% |
| 22 | ruff + pytest (ปัจจุบัน 30 เทสต์: การแปลงข้อมูล, schema, โมเดล, API, monitoring, | ruff + pytest (ปัจจุบัน 32 เทสต์: การแปลงข้อมูล, schema, โมเดล, API, monitoring, |
| 23 | รูปที่ 14 Job summary ของ model-quality: ด่าน roc_auc ไม่ผ่าน ส่วนด่านอื่นผ่าน | รูปที่ 14 Job summary ของ model-quality: ด่าน roc_auc ไม่ผ่าน ส่วนด่านอื่นผ่าน (ภาพก่อนเปลี่ยนด่าน latency จึงยังแสดงเกณฑ์ ≤ 25.0) |
| 24 | git clone github.com/super-SuPer-dev/loan-mlops && cd loan_mlops | git clone https://github.com/super-SuPer-dev/loan-mlops.git && cd loan-mlops |
| 25 | และ #9–#15) CI รันทุก PR และทุก PR ที่ merge ผ่านครบ 5/5 job | และ #9–#18) CI รันทุก PR และทุก PR ที่ merge ผ่านครบ 5/5 job |
| 26 | ชุดเทสต์มีทั้งหมด 38 ข้อ: 30 ข้อรันใน job code-quality (data 6, model 4, API 5, monitoring 6, data_prep 3, สคริปต์ test case 4 และรายงาน / Evidently 2) และอีก 8 ข้อใน tests/test_dags.py ซึ่งต้องมี Airflow จึงรันใน job dag-tests และถูกข้ามอัตโนมัติบนเครื่องที่ไม่มี Airflow เมื่อรันทั้งชุดบน / เครื่องที่ไม่มี Airflow ผ่าน 30 ข้อ (ข้าม 1 ไฟล์) | ชุดเทสต์มีทั้งหมด 40 ข้อ: 32 ข้อรันใน job code-quality (data 6, model 6, API 5, monitoring 6, data_prep 3, สคริปต์ test case 4 และรายงาน Evidently 2) และอีก 8 ข้อใน tests/test_dags.py ซึ่งต้องมี Airflow จึงรันใน job dag-tests และถูกข้ามอัตโนมัติบนเครื่องที่ไม่มี Airflow เมื่อรันทั้งชุดบนเครื่องที่ไม่มี Airflow ผ่าน 32 ข้อ (ข้าม 1 ไฟล์) |
| 26 | coverage ของ src/ รวม 85% (494 statements พลาด 73) โดย config.py, features.py และ monitoring.py ได้ 100%, drift_report.py และ / validation.py ได้ 94% และ steps.py ได้ 89% | coverage ของ src/ รวม 86% (519 statements พลาด 71) โดย config.py, features.py และ monitoring.py ได้ 100%, drift_report.py และ validation.py ได้ 94% และ steps.py ได้ 91% |
| 27 | และเพิ่ม unit test จน coverage ใน CI ถึง 85% | และเพิ่ม unit test จน coverage ใน CI ถึง 86% |
| 28 | Flask dev server ทำงานโปรเซสเดียว ความจุสูงสุดประมาณ 75 คำขอ/วินาที | Flask dev server ทำงานโปรเซสเดียว ความจุสูงสุดประมาณ 39 คำขอ/วินาที |
## ที่มาของตัวเลขใหม่

| ตัวเลข | มาจาก |
|---|---|
| ด่าน latency: p95 ที่ 20 คำขอ/วินาที ≤ 100 ms | `src/config.py` (`MAX_P95_LATENCY_MS = SLO_P95_MS / 2`) หลัง PR #16 |
| p50 20.7 / p95 54.6 ms, stress 10 ผู้ใช้ 261 / 314 ms 39 คำขอ/วินาที, stress 40 ผู้ใช้ 1,133 / 1,281 ms 36.3 คำขอ/วินาที | `scripts/load_test.py` วัดเมื่อ 6 ต.ค. ขณะเสิร์ฟ v2 LogReg (threshold 0.72) |
| alert firing ที่ 1.69 วินาที | ภาพหน้า Alerts ของ Prometheus ตอนกฎ firing (สไลด์หน้า 27) |
| v3 = logreg, test AUC 0.8898, threshold 0.55, W4 AUC 0.919 | `scripts/demo.py` รอบ 6 ต.ค. หลังแก้ด่าน latency (ได้ผลเดิมอีกครั้งบน clone ใหม่) |
| HistGradientBoosting 16.7 ms ทีละคำขอ, 147 ms ที่ 20 คำขอ/วินาที, p95 ของ API 222 ms | log ของ `select_best` และ load test ก่อนแก้ |
| เทสต์ 32 + 8 ข้อ, coverage 86% (519 statements พลาด 71) | `uv run pytest -q --cov=src` บน main เมื่อ 7 ต.ค. |
| hyperparameter ในตารางสไลด์หน้า 17 | `src/features.py` และ `notebooks/_build_notebook.py` |

## จุดที่ยังไม่ได้แก้

- **ภาพหลักฐาน CI (สไลด์หน้า 29, 30 และรายงานรูปที่ 12–14):** ตาราง model-quality ในภาพยังแสดงเกณฑ์เดิม `p95 latency ms (≤ 25.0)` เพราะเป็นภาพของ run ก่อนเปลี่ยนด่าน ฉบับแก้เพิ่มหมายเหตุกำกับไว้ ไม่ได้เปลี่ยนภาพ
- **สารบัญของรายงาน:** ไม่มีหัวข้อ 7.6 ที่เพิ่มใหม่ในหน้า 17
- **ตัวเลขจาก notebook** (threshold 0.77, ตัดสินอัตโนมัติ 78.5%, latency 19.74 / 27.74 ms): คงไว้ตามเดิม เพราะสไลด์และรายงานระบุว่าเป็นค่าจาก notebook
- **สไลด์หน้า 27 และรายงานรูปที่ 11** (Grafana แยกตามสถานะการจ้างงาน 94.9% / 62.1%): เป็นภาพเดิมที่ถ่ายช่วงมี traffic กลุ่ม W2 ตัวเลขตรงกับข้อความในรายงาน จึงคงไว้
