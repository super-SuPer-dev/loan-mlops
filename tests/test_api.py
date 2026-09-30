"""ทดสอบ Flask API ด้วย test client — ใส่โมเดลเล็ก ๆ เข้า STATE ตรง ๆ ไม่ต้องมี MLflow server"""
import pytest
from sklearn.linear_model import LogisticRegression

from serving import app as api
from src import features as F
from src import validation as V


@pytest.fixture()
def client(loans):
    api.STATE.update(model=None, schema=None, version=None)
    yield api.app.test_client()


@pytest.fixture()
def loaded(client, loans):
    model = F.build_pipeline(LogisticRegression(max_iter=1000)).fit(loans, F.encode_target(loans[F.TARGET]))
    api.STATE.update(model=model, schema=V.build_schema(loans, target=F.TARGET), version="1", threshold=0.5)
    return client


def request_body(loans):
    return loans.drop(columns=[F.TARGET]).iloc[0].to_dict()


def test_health_before_model_loaded(client):
    r = client.get("/health")
    assert r.status_code == 503 and r.json["model_loaded"] is False


def test_predict_ok(loaded, loans):
    r = loaded.post("/predict", json=request_body(loans))
    assert r.status_code == 200
    assert r.json["decision"] in {"AUTO_APPROVE", "AUTO_REJECT", "MANUAL_REVIEW"}
    assert 0 <= r.json["probability_approve"] <= 1


def test_predict_rejects_bad_request(loaded, loans):
    r = loaded.post("/predict", json={**request_body(loans), "Credit_Score": 999})
    assert r.status_code == 400 and "Credit_Score" in " ".join(r.json["details"])
    assert loaded.post("/predict", data="not json").status_code == 400


def test_batch_and_metrics(loaded, loans):
    rows = loans.drop(columns=[F.TARGET]).head(5).to_dict(orient="records")
    r = loaded.post("/predict/batch", json=rows)
    assert r.status_code == 200 and r.json["n"] == 5
    text = loaded.get("/metrics").data.decode()
    assert "loan_api_requests_total" in text and "loan_api_latency_seconds" in text
