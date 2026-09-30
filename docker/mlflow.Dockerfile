# syntax=docker/dockerfile:1
# MLflow tracking server + model registry (Lab 9 ใช้ไฟล์ SQLite บนเครื่อง ตรงนี้ยกเป็น server ให้ทุก container ใช้ร่วมกัน)
FROM python:3.12-slim

RUN --mount=type=cache,id=pip,target=/tmp/pipcache,uid=50000 \
    PIP_CACHE_DIR=/tmp/pipcache pip install mlflow==3.16.1
EXPOSE 5000
