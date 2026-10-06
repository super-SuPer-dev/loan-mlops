# คู่มือรันระบบและทดสอบ (จากเครื่องเปล่า)

ทดสอบขั้นตอนทั้งหมดในไฟล์นี้บน clone ใหม่แล้วเมื่อ 6 ต.ค. 2569 (Windows 11, Docker Desktop)

สารบัญ
1. [ติดตั้งและเปิดระบบ](#1-ติดตั้งและเปิดระบบ)
2. [รัน demo ทั้งระบบ](#2-รัน-demo-ทั้งระบบ)
3. [เปิดดูอะไรที่ไหน](#3-เปิดดูอะไรที่ไหน)
4. [ส่งข้อมูล 1 sample เข้า API](#4-ส่งข้อมูล-1-sample-เข้า-api)
5. [ทำข้อมูลให้เสียแล้วส่งมาทดสอบ](#5-ทำข้อมูลให้เสียแล้วส่งมาทดสอบ)
6. [ทดสอบด้วยไฟล์ CSV หลายแถว](#6-ทดสอบด้วยไฟล์-csv-หลายแถว)
7. [ปิดระบบ / ล้างระบบ](#7-ปิดระบบ--ล้างระบบ)
8. [ปัญหาที่เจอบ่อย](#8-ปัญหาที่เจอบ่อย)

## 1. ติดตั้งและเปิดระบบ

ต้องมี: Git, Docker Desktop (เปิดไว้ก่อน) และ [uv](https://docs.astral.sh/uv/)
พื้นที่ดิสก์ประมาณ 10 GB (image ของ Docker 9 GB + `.venv` 1 GB) และ RAM ที่ให้ Docker อย่างน้อย 6 GB

```bash
git clone https://github.com/super-SuPer-dev/loan-mlops.git
cd loan-mlops
uv sync
docker compose up -d --build
```

บน Windows ให้ clone ไว้ที่ path สั้น ๆ เช่น `C:\work\loan-mlops` (ถ้า path ยาวเกิน 260 ตัวอักษร `uv sync` จะติดตั้ง scikit-learn ไม่ครบ)

| ขั้น | เวลาโดยประมาณ | เกิดอะไรขึ้น |
|---|---|---|
| `uv sync` | 30 วินาที | ติดตั้งไลบรารี Python ลง `.venv` (ใช้รันสคริปต์และเทสต์บนเครื่อง) |
| `docker compose up -d --build` ครั้งแรก | 5–15 นาที ตามความเร็วเน็ต | build image 3 ตัว, สร้างฐานข้อมูล Airflow, ดาวน์โหลดข้อมูลจาก Kaggle ลง `data/` |
| ครั้งถัดไป | 30 วินาที | ใช้ image เดิม |

ตรวจว่าพร้อม:

```bash
docker compose ps
curl http://localhost:8000/health
```

`docker compose ps` ต้องเห็น 9 บริการสถานะ `Up` (`airflow-init` จะ `Exited (0)` ซึ่งถูกต้อง เพราะทำงานครั้งเดียวแล้วจบ)

`/health` ตอนยังไม่เคยเทรนจะตอบ **503** พร้อม `"status":"loading"` ซึ่งถูกต้อง เพราะยังไม่มี model ใน registry
หลังเทรนรอบแรกจะตอบ **200** พร้อม `"status":"ok"` และ `model_version`

ตรวจโค้ด (ไม่ต้องเปิด Docker):

```bash
uv run ruff check .
uv run pytest -q
```

ผลที่ควรได้: `All checks passed!` และ `32 passed, 1 skipped`

## 2. รัน demo ทั้งระบบ

```bash
uv run python scripts/demo.py
```

ใช้เวลาประมาณ 8 นาที รันครบ 7 ขั้นตามลำดับ

| # | ขั้น | ผลที่ควรเห็นใน terminal |
|---|---|---|
| 1 | เทรนรอบแรก | `success`, API โหลด version ใหม่ (LogReg, AUC ≈ 0.899, threshold 0.72) |
| 2 | ข้อมูลเสีย | `failed` (ตั้งใจ), ALERT `example_validator` ล้ม |
| 3 | W1 ปกติ | `drift_share=0.0`, `retrain=False`, ไม่มี alert |
| 4 | W2 data drift | `drift_share=0.63`, ALERT data drift แต่ `retrain=False` |
| 5 | W3 concept drift | AUC ตกเหลือ ≈ 0.80, `retrain=True`, ได้ version ใหม่ (AUC ≈ 0.890, threshold 0.55) |
| 6 | W4 หลัง retrain | AUC กลับมา ≈ 0.919, ไม่มี alert |
| 7 | Load test | p95 ต่ำกว่า 200 ms, `✅ ผ่าน SLO` |

รันเฉพาะบางขั้น: `uv run python scripts/demo.py --only 2 5`

คำสั่งเสริม:

| ต้องการ | คำสั่ง |
|---|---|
| Rollback ไป `@previous` | `uv run python scripts/rollback.py` หรือ Trigger `loan_rollback_pipeline` ใน Airflow |
| เทสต์ DAG ใน container | `docker compose exec scheduler python -m pytest /opt/project/tests/test_dags.py -q` |
| Load test อย่างเดียว | `uv run python scripts/load_test.py` |
| ทำให้ alert latency firing | `uv run python scripts/load_test.py --stress --n 3000 --concurrency 40` |

บน Windows ถ้าตัวอักษรไทยใน terminal ขึ้น error ให้ตั้งค่าก่อนรัน: PowerShell ใช้ `$env:PYTHONUTF8 = "1"`, Git Bash ใช้ `export PYTHONUTF8=1`

## 3. เปิดดูอะไรที่ไหน

ทุกหน้าไม่ต้องล็อกอิน

| หน้า | URL | กดตรงไหน | ดูอะไร |
|---|---|---|---|
| Airflow | http://localhost:8080 | **Dags** → คลิกชื่อ DAG → คลิก run ในแถบซ้าย → แท็บ **Graph** | task สีเขียว (สำเร็จ) / แดง (ล้ม) / ชมพู (ข้าม) |
| Airflow: log | เดียวกัน | คลิก task ใน Graph → แท็บ **Logs** | เหตุผลที่ task ล้ม เช่น รายการปัญหาของข้อมูลเสีย |
| Airflow: สั่งรัน | เดียวกัน | เปิดสวิตช์หน้าชื่อ DAG (unpause) → ปุ่ม **Trigger** มุมขวาบน | เทรนใหม่ / monitoring / rollback |
| MLflow: การทดลอง | http://localhost:5050 | **Experiments** → `loan-approval` → ติ๊ก run ที่ต้องการ → **Compare** | param, metric ของแต่ละ run |
| MLflow: registry | เดียวกัน | **Models** → `loan-approval` | ทุก version, alias `@champion` / `@previous`, tag สถานะ |
| API | http://localhost:8000/health | เปิดใน browser ได้เลย | version ที่ให้บริการอยู่ |
| API: รายละเอียด model | http://localhost:8000/model | เปิดใน browser | threshold, AUC, alert ล่าสุด |
| Prometheus | http://localhost:9090/alerts | คลิกชื่อกฎเพื่อขยาย | กฎ 7 ข้อ สถานะ inactive / pending / firing |
| Grafana | http://localhost:3000 | เปิดแล้วเจอ dashboard เลย | latency, ผล cascade, สัดส่วนปฏิเสธแยกกลุ่ม, drift, AUC |

DAG มี 3 ตัว: `loan_training_pipeline`, `loan_monitoring_pipeline`, `loan_rollback_pipeline`

ไฟล์ผลลัพธ์บนเครื่อง:

| ไฟล์ | เนื้อหา |
|---|---|
| `include/alerts.jsonl` | alert ทุกครั้ง (ข้อมูลเสีย, drift, promote, rollback) |
| `include/schema.json` | schema ที่ระบบเรียนจากข้อมูล train |
| `include/monitoring/` | ผลตรวจ drift ของแต่ละ batch |
| `reports/load_test.json` | ผล load test ล่าสุด |
| `reports/test_cases.json` | ผลของ `run_test_cases.py` ล่าสุด |
| `logs/api/` | log คำขอและผลทำนายของ API |

## 4. ส่งข้อมูล 1 sample เข้า API

ต้องมี model ก่อน (`/health` ได้ `"status":"ok"`) ถ้ายังไม่มีให้รัน `uv run python scripts/demo.py --only 1`

ไฟล์ตัวอย่างที่เตรียมไว้:

| ไฟล์ | ใช้ทำอะไร |
|---|---|
| [scripts/sample_request.json](../scripts/sample_request.json) | คำขอสินเชื่อ 1 รายการที่ถูกต้อง (28 ฟิลด์) |
| [scripts/sample_bad_request.json](../scripts/sample_bad_request.json) | รายการเดียวกันที่ทำให้เสีย 2 จุด (`Credit_Score` = 920, `Employment_Status` = Freelancer) |
| [scripts/sample_request.csv](../scripts/sample_request.csv) | รายการเดียวกันในรูป CSV 1 แถว (เปิดแก้ใน Excel ได้) |

### ส่งด้วย curl

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @scripts/sample_request.json
```

ใน PowerShell ให้พิมพ์ `curl.exe` แทน `curl`

ผลที่ได้ (HTTP 200):

```json
{"approved": false, "decision": "AUTO_REJECT", "model_version": "3", "probability_approve": 0.2854}
```

| ฟิลด์ | ความหมาย |
|---|---|
| `probability_approve` | ความน่าจะเป็นที่ model ให้ว่าควรอนุมัติ |
| `decision` | ผล cascade: `AUTO_APPROVE` / `AUTO_REJECT` / `MANUAL_REVIEW` (ก้ำกึ่ง ส่งให้เจ้าหน้าที่) |
| `approved` | `true` เมื่อความน่าจะเป็นถึง threshold |
| `model_version` | version ใน registry ที่ตอบคำขอนี้ |

ค่าความน่าจะเป็นขึ้นกับ version ของ model ตัวเลขในไฟล์นี้มาจาก version 3 (threshold 0.55)

ส่งข้อมูลเสีย:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @scripts/sample_bad_request.json
```

ผลที่ได้ (HTTP 400) บอกทุกจุดที่ผิด:

```json
{
    "details": [
        "Credit_Score: ผิดกฎโดเมน ต้องอยู่ใน [300, 850] แต่พบ 920 ถึง 920",
        "Employment_Status: พบหมวดที่ไม่เคยเห็น ['Freelancer']"
    ],
    "error": "ข้อมูลไม่ผ่าน schema"
}
```

curl แสดงภาษาไทยเป็นรหัส `\u0e..` ถ้าต้องการอ่านเป็นภาษาไทยให้ต่อท้ายคำสั่งด้วย `| uv run python -m json.tool --no-ensure-ascii`
ถ้าต้องการเห็นเลข status ให้เพิ่ม `-i` หลัง `curl`

### ส่งด้วยสคริปต์ (อ่านง่ายกว่า)

```bash
uv run python scripts/run_test_cases.py --file scripts/sample_request.csv
```

```
[1] ตรวจทั้งไฟล์กับ schema: ผ่าน ✅
[2] ผลต่อแถว (แสดง 1 แถวแรก)
   แถว    0: 200  AUTO_REJECT    p=0.2854
[3] สรุป: {"n_rows": 1, "status_counts": {"200": 1}, "decision_counts": {"AUTO_REJECT": 1}}
```

### ส่งหลายรายการในคำขอเดียว

`POST /predict/batch` รับ JSON array ของคำขอ (ส่ง array เข้า `/predict` จะได้ 400)

## 5. ทำข้อมูลให้เสียแล้วส่งมาทดสอบ

เปิด `scripts/sample_request.json` (หรือ `.csv`) แก้ค่าตามตาราง แล้วส่งด้วยคำสั่งในข้อ 4
ผลในตารางได้จากการส่งจริง โดยแก้ทีละจุดจากไฟล์ตัวอย่าง

### การแก้ที่ระบบปฏิเสธ (HTTP 400)

| แก้อะไร | ตัวอย่าง | ข้อความที่ API ตอบ |
|---|---|---|
| ค่าเกินกฎโดเมน | `"Credit_Score": 920` | `Credit_Score: ผิดกฎโดเมน ต้องอยู่ใน [300, 850]` |
| ค่าติดลบ | `"Annual_Income": -5000` | `Annual_Income: พบค่าติดลบ -5000 ทั้งที่ห้ามติดลบ` |
| หมวดที่ไม่เคยเห็น | `"Employment_Status": "Freelancer"` | `Employment_Status: พบหมวดที่ไม่เคยเห็น ['Freelancer']` |
| ลบฟิลด์ออก | ลบบรรทัด `"Collateral"` | `ขาดคอลัมน์ที่ schema กำหนด: ['Collateral']` |
| ใส่ข้อความในช่องตัวเลข | `"Age": "abc"` | `Age: มีค่าที่ไม่ใช่ตัวเลข 1 แถว` |
| ค่าไกลจากช่วงที่เคยเห็นมาก | `"Loan_Amount": 5000000` | `Loan_Amount: ค่าอยู่นอกช่วงที่คาด` |
| รายได้ต่อเดือนไม่ตรงกับรายปี | `"Monthly_Income": 100` | `Monthly_Income ไม่เท่ากับ Annual_Income/12` |

### การแก้ที่ระบบยังรับ (HTTP 200)

| แก้อะไร | ตัวอย่าง | ผล | เหตุผล |
|---|---|---|---|
| ค่าว่างในฟิลด์เดียว | `"Savings_Balance": null` | 200 ทำนายได้ | model เติมค่ากลางให้ ส่วนเกณฑ์ค่าว่างเกิน 5% ใช้ตรวจทั้ง batch ใน pipeline |
| เพิ่มฟิลด์ที่ไม่รู้จัก | `"Foo": 1` | 200 ทำนายได้ | ฟิลด์เกินถูกทิ้ง ไม่กระทบการทำนาย |
| เปลี่ยนเป็นผู้ขอที่ดีขึ้น | `"Credit_Score": 790, "Debt_To_Income_Ratio": 8, "Existing_Loans": 0` | 200 `AUTO_APPROVE` (p = 0.9149) | ข้อมูลถูกต้อง ผลทำนายเปลี่ยนตามค่า |
| เปลี่ยนเป็นผู้ขอก้ำกึ่ง | `"Credit_Score": 680` | 200 `MANUAL_REVIEW` (p = 0.5151) | ความน่าจะเป็นอยู่ใกล้ threshold |

ถ้าแก้ `Annual_Income` ต้องแก้ `Monthly_Income` ให้เท่ากับ `Annual_Income / 12` ด้วย ไม่อย่างนั้นจะได้ 400 จากกฎข้อสุดท้ายในตารางแรก

### ค่าที่ถูกต้องของแต่ละฟิลด์

| ฟิลด์ตัวเลข | ช่วงที่รับ |
|---|---|
| `Age` | 18–100 |
| `Credit_Score` | 300–850 |
| `Debt_To_Income_Ratio` | 0–100 |
| `Loan_Term_Months` | 1–480 |
| ฟิลด์จำนวนเงิน จำนวนปี จำนวนครั้ง | ห้ามติดลบ และต้องไม่ไกลจากช่วงในข้อมูล train เกิน 20% ของความกว้างช่วง |

| ฟิลด์หมวดหมู่ | ค่าที่รับ |
|---|---|
| `Education_Level` | Bachelor, High School, Master, PhD |
| `Employment_Status` | Business Owner, Contract, Salaried, Self-Employed, Unemployed |
| `Loan_Purpose` | Business, Car, Debt Consolidation, Education, Home, Medical, Personal |
| `Property_Ownership` | Family Owned, Mortgage, Own, Rent |
| `Residential_Status` | Rural, Suburban, Urban |
| `Co_Applicant`, `Collateral` | Yes, No |
| `Application_Channel` | Agent, Branch, Mobile App, Online |
| `Region` | Central, East, North, South, West |
| `Previous_Loan_Status` | Active, Defaulted, No Previous Loan, Paid Off |

ช่วงจริงของทุกฟิลด์อยู่ใน `include/schema.json` (สร้างหลังเทรนรอบแรก)

## 6. ทดสอบด้วยไฟล์ CSV หลายแถว

วางไฟล์ไว้ในโฟลเดอร์โปรเจกต์แล้วรัน:

```bash
uv run python scripts/run_test_cases.py --file <ชื่อไฟล์>.csv
```

สคริปต์ทำ 3 อย่าง: ตรวจทั้งไฟล์กับ schema, ส่งทีละแถวเข้า `/predict`, และสรุปผล
ถ้าไฟล์มีคอลัมน์ `Loan_Approved` (เฉลย) จะคำนวณ accuracy และ ROC-AUC ให้

ไฟล์ที่มีอยู่แล้ว (สร้างตอน `docker compose up` ครั้งแรก):

| ไฟล์ | ผลที่ได้ |
|---|---|
| `data/production/week_1_normal.csv` | 200 ทั้ง 1,500 แถว |
| `data/production/bad_data.csv` | schema พบ 6 ปัญหา, 400 ทั้ง 2,000 แถวพร้อมเหตุผล |

**ข้อมูลปกติควรทดสอบกับ model รอบแรก** หลัง demo ขั้น 5 model ถูกเทรนใหม่ตามนโยบายอนุมัติแบบใหม่ ถ้านำข้อมูลนโยบายเดิมมาวัดจะได้ AUC ต่ำ (`week_1_normal.csv` ได้ 0.762 บน model หลัง retrain เทียบกับ 0.895 บน model รอบแรก) เลือกทางใดทางหนึ่ง:

```bash
uv run python scripts/reset_demo.py --yes
uv run python scripts/demo.py --only 1
```

หรือ rollback หลัง demo:

```bash
uv run python scripts/rollback.py
```

### ทดสอบผ่าน pipeline แทน API

1. วางไฟล์ในโฟลเดอร์ `data/`
2. Airflow → `loan_training_pipeline` → **Trigger** → ใส่ config `{"data_file": "<ชื่อไฟล์>"}`
3. ข้อมูลเสีย: task `example_validator` เป็นสีแดง pipeline หยุด และมี alert ใน `include/alerts.jsonl`
4. ข้อมูลปกติ: เทรนและผ่านด่านตรวจตามปกติ

## 7. ปิดระบบ / ล้างระบบ

| ต้องการ | คำสั่ง | ผล |
|---|---|---|
| ปิดชั่วคราว | `docker compose down` | container หยุด ข้อมูล (registry, ประวัติ run, metric) ยังอยู่ |
| เปิดกลับ | `docker compose up -d` | กลับมาสถานะเดิม |
| ล้างแล้วเริ่มใหม่ | `uv run python scripts/reset_demo.py --yes` | ลบ registry, ฐานข้อมูล Airflow, metric, `include/`, `logs/`, `reports/` แล้วเปิดระบบใหม่ (เก็บไฟล์จาก Kaggle ไว้) |
| ลบทุกอย่าง | `docker compose down -v` | ลบ container และ volume ทั้งหมด |

## 8. ปัญหาที่เจอบ่อย

| อาการ | สาเหตุ | แก้ |
|---|---|---|
| `/health` ได้ 503 | ยังไม่มี model | `uv run python scripts/demo.py --only 1` หรือ Trigger `loan_training_pipeline` |
| pytest พังที่ `sklearn ... _argkmin` | path ของโฟลเดอร์ยาวเกิน | ย้ายไป path สั้น แล้ว `uv sync` ใหม่ |
| `UnicodeEncodeError` ตอนพิมพ์ภาษาไทย | terminal ของ Windows ไม่ใช่ UTF-8 | ตั้ง `PYTHONUTF8=1` |
| `port is already allocated` | มีโปรแกรมอื่นใช้พอร์ต 8080 / 5050 / 8000 / 9090 / 3000 | ปิดโปรแกรมนั้น หรือ `docker compose down` ของโปรเจกต์อื่น |
| Airflow ไม่เห็น DAG | dag-processor ยังไม่อ่านไฟล์ | `docker compose restart dag-processor scheduler` |
| `docker compose exec ... /opt/project/...` หา path ไม่เจอใน Git Bash | Git Bash แปลง path | `export MSYS_NO_PATHCONV=1` ก่อนรัน |
| สลับ git branch แล้วระบบเปลี่ยนพฤติกรรม | โฟลเดอร์โปรเจกต์ถูก mount เข้า container | ให้เครื่องสาธิตอยู่บน `main` |
