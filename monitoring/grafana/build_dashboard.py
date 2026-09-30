"""สร้างไฟล์ dashboard ของ Grafana (JSON) จากรายการ panel ที่อ่านง่าย

    uv run python monitoring/grafana/build_dashboard.py

แก้ panel ที่ไฟล์นี้แล้วรันใหม่ ไม่ต้องแก้ JSON ยาว ๆ ด้วยมือ
Grafana โหลดไฟล์ผลลัพธ์อัตโนมัติจาก provisioning/dashboards/
"""
import json
from pathlib import Path

OUT = Path(__file__).parent / "provisioning" / "dashboards" / "loan_dashboard.json"
DS = {"type": "prometheus", "uid": "prometheus"}

P95 = 'histogram_quantile({q}, sum by (le) (rate(loan_api_latency_seconds_bucket{{endpoint="/predict"}}[1m])))'
# "or vector(0)" = ถ้ายังไม่เคยมี 5xx เลย (ไม่มี series) ให้ถือว่าเป็น 0 แทนการขึ้น No data
ERR = ('(sum(rate(loan_api_requests_total{status=~"5.."}[1m])) or vector(0))'
       ' / sum(rate(loan_api_requests_total[1m]))')
REJECT_BY_JOB = ('sum by (employment_status) (increase(loan_api_decisions_by_employment_total{decision="AUTO_REJECT"}[1h]))'
                 ' / sum by (employment_status) (increase(loan_api_decisions_by_employment_total[1h]))')
REJECT = ('sum(rate(loan_api_decisions_total{decision="AUTO_REJECT"}[5m])) '
          '/ sum(rate(loan_api_decisions_total[5m]))')


def target(expr, legend="", ref="A"):
    return {"datasource": DS, "expr": expr, "legendFormat": legend, "refId": ref}


def thresholds(*steps):
    """steps = [(ค่า, สี), ...] ค่าแรกเป็น None = ค่าตั้งต้น"""
    return {"mode": "absolute", "steps": [{"value": v, "color": c} for v, c in steps]}


def stat(title, expr, x, y, w=4, unit="none", th=None, desc=""):
    return {"type": "stat", "title": title, "description": desc, "datasource": DS,
            "gridPos": {"x": x, "y": y, "w": w, "h": 4}, "targets": [target(expr)],
            "fieldConfig": {"defaults": {"unit": unit, "thresholds": th or thresholds((None, "blue"))},
                            "overrides": []},
            "options": {"colorMode": "background", "graphMode": "area", "reduceOptions": {"calcs": ["lastNotNull"]}}}


def series(title, targets, x, y, w=12, h=8, unit="none", th=None, stack=False, desc=""):
    custom = {"lineWidth": 2, "fillOpacity": 30 if stack else 10,
              "stacking": {"mode": "normal" if stack else "none"},
              "thresholdsStyle": {"mode": "line" if th else "off"}}
    return {"type": "timeseries", "title": title, "description": desc, "datasource": DS,
            "gridPos": {"x": x, "y": y, "w": w, "h": h}, "targets": targets,
            "fieldConfig": {"defaults": {"unit": unit, "custom": custom,
                                         "thresholds": th or thresholds((None, "green"))}, "overrides": []},
            "options": {"legend": {"displayMode": "list", "placement": "bottom"}, "tooltip": {"mode": "multi"}}}


def row(title, y):
    return {"type": "row", "title": title, "collapsed": False, "gridPos": {"x": 0, "y": y, "w": 24, "h": 1},
            "panels": []}


panels = [
    row("สถานะระบบ (System health) — SLO: p95 ≤ 200 ms ที่ 20 req/s, 5xx ≤ 1%", 0),
    stat("โมเดลที่ให้บริการ (version)", "loan_api_model_version", 0, 1),
    stat("API พร้อมให้บริการ", "min(up{job=\"loan-api\"}) * min(loan_api_model_loaded)", 4, 1,
         th=thresholds((None, "red"), (1, "green")), desc="1 = API ตอบและโหลดโมเดลแล้ว"),
    stat("คำขอ / วินาที", 'sum(rate(loan_api_requests_total{endpoint="/predict"}[1m]))', 8, 1, unit="reqps"),
    stat("p95 latency", P95.format(q=0.95), 12, 1, unit="s",
         th=thresholds((None, "green"), (0.15, "orange"), (0.2, "red"))),
    stat("5xx error rate", ERR, 16, 1, unit="percentunit",
         th=thresholds((None, "green"), (0.005, "orange"), (0.01, "red"))),
    stat("คำขอผิดรูปแบบ (400) / นาที", 'sum(increase(loan_api_requests_total{status="400"}[1m]))', 20, 1,
         desc="คำขอที่ไม่ผ่าน schema — ถ้าพุ่งแปลว่าระบบต้นทางส่งข้อมูลผิด"),
    series("Latency ของ /predict เทียบ SLO", [
        target(P95.format(q=0.50), "p50", "A"), target(P95.format(q=0.95), "p95", "B"),
        target(P95.format(q=0.99), "p99", "C")], 0, 5, unit="s",
        th=thresholds((None, "green"), (0.2, "red"))),
    series("คำขอแยกตาม endpoint และ status", [
        target("sum by (endpoint, status) (rate(loan_api_requests_total[1m]))", "{{endpoint}} {{status}}")],
        12, 5, unit="reqps", stack=True),

    row("การตัดสินใจของโมเดล (Prediction monitoring — ไม่ต้องรอเฉลย)", 13),
    series("ผลตัดสิน (cascade) ต่อวินาที", [
        target("sum by (decision) (rate(loan_api_decisions_total[1m]))", "{{decision}}")],
        0, 14, w=8, unit="reqps", stack=True),
    series("สัดส่วนปฏิเสธอัตโนมัติ (เตือนเมื่อ > 70%)", [target(REJECT, "AUTO_REJECT share")],
           8, 14, w=8, unit="percentunit", th=thresholds((None, "green"), (0.7, "red")),
           desc="prediction drift: ถ้าสัดส่วนเปลี่ยนมาก แปลว่าผู้สมัครหรือโมเดลเปลี่ยน"),
    {"type": "bargauge", "title": "สัดส่วนปฏิเสธอัตโนมัติ แยกตามสถานะการจ้างงาน (1 ชม.)", "datasource": DS,
     "description": "metric เฉพาะโจทย์: ถ้ากลุ่มใดถูกปฏิเสธพุ่งขึ้นผิดปกติ อาจเป็น drift หรือความไม่เป็นธรรม",
     "gridPos": {"x": 16, "y": 14, "w": 8, "h": 8},
     "targets": [target(REJECT_BY_JOB, "{{employment_status}}")],
     "fieldConfig": {"defaults": {"unit": "percentunit", "min": 0, "max": 1,
                                  "thresholds": thresholds((None, "green"), (0.6, "orange"), (0.8, "red"))},
                     "overrides": []},
     "options": {"orientation": "horizontal", "displayMode": "gradient",
                 "reduceOptions": {"calcs": ["lastNotNull"]}}},
    # heatmap แปลง bucket แบบสะสมของ Prometheus เป็นจำนวนต่อช่วงให้เอง (format=heatmap)
    {"type": "heatmap", "title": "การกระจายของความน่าจะเป็นที่ทำนาย (ต่อนาที)", "datasource": DS,
     "description": "ถ้ารูปทรงเปลี่ยนไปจากปกติ = prediction drift (เห็นได้ก่อนเฉลยมาถึง)",
     "gridPos": {"x": 0, "y": 22, "w": 24, "h": 7},
     "targets": [{**target("sum by (le) (increase(loan_api_probability_bucket[1m]))", "{{le}}"),
                  "format": "heatmap"}],
     "options": {"calculate": False, "yAxis": {"axisLabel": "ความน่าจะเป็นที่จะอนุมัติ"},
                 "color": {"scheme": "Oranges", "mode": "scheme"}, "cellGap": 1}},
    row("คุณภาพโมเดล (จาก monitoring DAG เมื่อเฉลยมาถึง)", 29),
    stat("Data drift share", "max(loan_monitor_drift_share)", 0, 30, w=6, unit="percentunit",
         th=thresholds((None, "green"), (0.2, "red")), desc="เกณฑ์ 20% (KS / Chi-square)"),
    stat("AUC บน batch ล่าสุด", "max(loan_monitor_roc_auc)", 6, 30, w=6, unit="none",
         th=thresholds((None, "red"), (0.85, "green"))),
    stat("AUC ตกจากตอน deploy", "max(loan_monitor_auc_drop)", 12, 30, w=6, unit="none",
         th=thresholds((None, "green"), (0.05, "red")), desc="เกิน 0.05 = concept drift → retrain อัตโนมัติ"),
    stat("batch ที่ตรวจล่าสุด", "max(loan_monitor_checked_at) * 1000", 18, 30, w=6, unit="dateTimeFromNow"),
    series("AUC ปัจจุบัน เทียบ AUC ตอน deploy", [
        target("max(loan_monitor_roc_auc)", "AUC batch ล่าสุด", "A"),
        target("max(loan_monitor_reference_roc_auc)", "AUC ตอน deploy", "B")], 0, 34, w=12, unit="none"),
    series("Data drift share เทียบเกณฑ์", [target("max(loan_monitor_drift_share)", "drift share")],
           12, 34, w=12, unit="percentunit", th=thresholds((None, "green"), (0.2, "red"))),
]

dashboard = {
    "uid": "loan-mlops", "title": "Loan Approval — Serving & Model Monitoring", "timezone": "browser",
    "schemaVersion": 39, "version": 1, "refresh": "5s", "time": {"from": "now-30m", "to": "now"},
    "tags": ["loan", "mlops"], "panels": [{**p, "id": i + 1} for i, p in enumerate(panels)],
}
OUT.write_text(json.dumps(dashboard, indent=2, ensure_ascii=False), encoding="utf-8")
print("wrote", OUT, len(panels), "panels")
