# syntax=docker/dockerfile:1
# Airflow + ไลบรารี ML (แบบ Dockerfile ของ Lab 11)
# ห้ามลง apache-airflow ซ้ำใน requirements เพราะ image มีอยู่แล้ว ลงซ้ำเวอร์ชันจะชนกัน
FROM apache/airflow:3.3.1-python3.12

COPY requirements-ml.txt /tmp/requirements-ml.txt
# cache ของ pip ใช้ร่วมกันทุก Dockerfile (id=pip) → ไฟล์ wheel ที่โหลดแล้วไม่ต้องโหลดซ้ำตอน build image อื่น
RUN --mount=type=cache,id=pip,target=/tmp/pipcache,uid=50000 \
    PIP_CACHE_DIR=/tmp/pipcache pip install -r /tmp/requirements-ml.txt

# สำหรับรัน tests/test_dags.py ภายใน container
RUN pip install pytest==9.1.1
