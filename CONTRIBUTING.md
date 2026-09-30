# วิธีทำงานร่วมกัน (ทุกคนต้องทำผ่าน Pull Request)

คะแนนรายบุคคลดูจากประวัติ commit, การประเมินกันเองในกลุ่ม และการตอบคำถามวันนำเสนอ
ทุกคนจึงต้องมี commit และ PR ของตัวเอง และ**ต้องอธิบายโค้ดที่ตัวเองส่งได้ทุกบรรทัด**

งานของแต่ละคนอยู่ใน [Issues](../../issues) (label `task`) ให้ assign ตัวเองก่อนเริ่มทำ

## ขั้นตอน

1. เริ่มจาก `main` ล่าสุดเสมอ
   ```bash
   git switch main
   git pull
   ```
2. สร้าง branch ตามรูปแบบ `feature/<ชื่อเล่น>-<งาน>` เช่น `feature/ploy-rollback-dag`
   ```bash
   git switch -c feature/ploy-rollback-dag
   ```
3. ทำงาน แล้วตรวจให้ผ่านบนเครื่องก่อน push
   ```bash
   uv sync
   uv run ruff check .
   uv run pytest -q
   ```
4. commit ด้วยข้อความที่บอกว่าทำอะไร (ภาษาไทยหรืออังกฤษก็ได้) แล้ว push
   ```bash
   git add <ไฟล์ที่แก้>
   git commit -m "เพิ่ม rollback DAG สั่งย้อนเวอร์ชันได้จากหน้าเว็บ Airflow"
   git push -u origin feature/ploy-rollback-dag
   ```
5. เปิด Pull Request เข้า `main` ใส่ `Closes #<เลข issue>` ในคำอธิบาย แล้วรอ CI ให้เขียวครบ
6. ให้เพื่อนอย่างน้อย 1 คน review และกด Approve ก่อน merge (ใช้ **Squash and merge**) แล้วลบ branch

## ข้อควรระวัง

- **อย่า merge PR #1** (หลักฐาน CI ไม่ผ่าน) และอย่า merge PR ที่ CI แดง
- เครื่องที่รัน `docker compose` อยู่จะใช้โค้ดใน working copy ทันที (โฟลเดอร์ถูก mount เข้า container)
  การสลับ branch บนเครื่องนั้นจึงเปลี่ยนระบบที่รันอยู่ด้วย ทำงานบนเครื่องอื่น หรือสลับกลับ `main` ก่อนสาธิต
- ห้าม commit ข้อมูล (`data/`), `mlflow.db`, `include/`, `logs/`, ไฟล์ `.env` หรือ token ใด ๆ (`.gitignore` กันไว้แล้ว)
- เพิ่มไลบรารีใหม่ต้องเพิ่มทั้ง `pyproject.toml` (`uv add`) และ `requirements-ml.txt` (ถ้า Airflow/API ใช้) และระบุเวอร์ชันตายตัว
- ใช้ AI ช่วยได้ตามกติกาวิชา แต่ต้องเขียนในคำอธิบาย PR ว่าใช้ช่วยส่วนไหน และต้องอธิบายโค้ดได้เอง
