# รัน test case ของอาจารย์ (วันนำเสนอ)

อาจารย์จะให้ข้อมูลมาทดสอบทั้ง **กรณีปกติ** และ **กรณีผิดปกติ** ให้วางไฟล์ไว้ในโฟลเดอร์โปรเจกต์ แล้วรัน

```bash
uv run python scripts/run_test_cases.py --file <ไฟล์ของอาจารย์>.csv
```

ต้องเปิดระบบไว้ก่อน (`docker compose up -d`) และต้องมีโมเดล champion แล้ว (`curl http://localhost:8000/health` ได้ `"status":"ok"`)

## อ่านผลอย่างไร

| ส่วน | ความหมาย |
|---|---|
| [1] ตรวจทั้งไฟล์กับ schema | ผ่าน = ข้อมูลหน้าตาถูกต้อง · พบปัญหา = บอกว่าคอลัมน์ไหนผิดอย่างไร (เกินช่วง, ติดลบ, หมวดใหม่, ค่าว่างเกิน 5%, คอลัมน์หาย) |
| [2] ผลต่อแถว | `200` + `AUTO_APPROVE` / `AUTO_REJECT` / `MANUAL_REVIEW` = ทำนายได้ · `400` + เหตุผล = API ปฏิเสธข้อมูลผิด **ซึ่งเป็นพฤติกรรมที่ถูกต้อง** |
| [3] สรุป | จำนวนแต่ละ status, จำนวนแต่ละการตัดสิน, และถ้าไฟล์มี `Loan_Approved`: accuracy + ROC-AUC |

ผลเต็มอยู่ที่ `reports/test_cases.json`

## ลองกับข้อมูลที่มีอยู่แล้ว

```bash
uv run python scripts/run_test_cases.py --file data/production/week_1_normal.csv
uv run python scripts/run_test_cases.py --file data/production/bad_data.csv
```

## ถ้าอาจารย์ให้ทดสอบผ่าน pipeline แทน API

Trigger `loan_training_pipeline` พร้อม config `{"data_file": "<ชื่อไฟล์ในโฟลเดอร์ data/>"}`
ข้อมูลเสียจะทำให้ task `example_validator` ล้มและระบบแจ้งเตือน ข้อมูลปกติจะเทรนและผ่านด่านตรวจตามปกติ
