# syntax=docker/dockerfile:1
# Flask API สำหรับให้บริการโมเดล (build จากรากโปรเจกต์: docker build -f serving/Dockerfile .)
FROM python:3.12-slim

WORKDIR /app
COPY requirements-ml.txt serving/requirements.txt ./
RUN --mount=type=cache,id=pip,target=/tmp/pipcache,uid=50000 \
    PIP_CACHE_DIR=/tmp/pipcache pip install -r requirements-ml.txt -r requirements.txt

# โค้ดการแปลงข้อมูล/ตรวจ schema ชุดเดียวกับตอนเทรน (กัน Training-Serving Skew)
COPY src/ src/
COPY serving/app.py serving/app.py

ENV ML_PROJECT_ROOT=/app PYTHONUNBUFFERED=1 PORT=8000
EXPOSE 8000
# Docker ตรวจสุขภาพเองทุก 15 วินาที ผ่าน /health
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"
CMD ["python", "serving/app.py"]
