"""ทดสอบสคริปต์ run_test_cases ด้วย Flask test client (ไม่ต้องมี API จริงหรือ MLflow)"""
import sys
from pathlib import Path

import pytest
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_test_cases as rtc  # noqa: E402

from serving import app as api  # noqa: E402
from src import features as F  # noqa: E402
from src import validation as V  # noqa: E402


@pytest.fixture()
def post(loans):
    """ฟังก์ชัน post ที่คุยกับ Flask app ในโปรเซส (ใส่โมเดลเล็ก ๆ ลง STATE ตรง ๆ)"""
    model = F.build_pipeline(LogisticRegression(max_iter=1000)).fit(loans, F.encode_target(loans[F.TARGET]))
    api.STATE.update(model=model, schema=V.build_schema(loans, target=F.TARGET), version="1", threshold=0.5)
    client = api.app.test_client()

    def _post(payload):
        r = client.post("/predict", json=payload)
        return r.status_code, r.get_json()
    return _post


def test_normal_rows_are_predicted_and_scored(loans, post):
    report = rtc.run(loans.head(40), post)
    s = report["summary"]
    assert s["status_counts"] == {"200": 40}
    assert 0.0 <= s["accuracy"] <= 1.0 and 0.0 <= s["roc_auc"] <= 1.0
    assert sum(s["decision_counts"].values()) == 40


def test_bad_rows_get_400_with_reasons(loans, post):
    df = loans.head(5).copy()
    df.loc[df.index[0], "Credit_Score"] = 999            # นอกช่วง 300–850
    df.loc[df.index[1], "Employment_Status"] = "Astronaut"  # หมวดที่ไม่เคยเห็น
    report = rtc.run(df, post)
    assert report["summary"]["status_counts"] == {"200": 3, "400": 2}
    reasons = " ".join(" ".join(r["errors"]) for r in report["rows"] if r["status"] == 400)
    assert "Credit_Score" in reasons and "Astronaut" in reasons


def test_file_without_label_still_works(loans, post):
    report = rtc.run(loans.head(10).drop(columns=[F.TARGET]), post)
    assert report["summary"]["status_counts"] == {"200": 10}
    assert "accuracy" not in report["summary"]           # ไม่มีเฉลย ก็ไม่คำนวณ


def test_missing_values_are_sent_as_null(loans, post):
    df = loans.head(3).copy()
    df.loc[df.index[0], "Savings_Balance"] = float("nan")  # ค่าว่างรายแถว → imputer เติมให้
    report = rtc.run(df, post)
    assert report["summary"]["status_counts"] == {"200": 3}
