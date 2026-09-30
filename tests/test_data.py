"""ด่านความถูกต้องของข้อมูล: schema ต้องจับข้อมูลเสียได้ทุกแบบ และการแปลงข้อมูลต้องทนค่าว่าง/ค่าแปลก"""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src import features as F
from src import validation as V
from src.data_prep import make_bad_batch


def test_good_data_passes(loans):
    schema = V.build_schema(loans, target=F.TARGET)
    assert V.validate(loans, schema) == []


def test_bad_data_is_caught_and_stops(loans):
    schema = V.build_schema(loans, target=F.TARGET)
    bad = make_bad_batch(loans)
    problems = " | ".join(V.validate(bad, schema))
    for expected in ["Collateral", "ติดลบ", "Credit_Score", "ค่าว่าง", "Freelancer", "Monthly_Income"]:
        assert expected in problems, f"ไม่จับ: {expected}"
    with pytest.raises(V.DataValidationError):
        V.validate_or_raise(bad, schema)


def test_single_request_allows_missing_but_not_invalid(loans):
    schema = V.build_schema(loans, target=F.TARGET)
    row = loans.drop(columns=[F.TARGET]).iloc[[0]].copy()
    row["Savings_Balance"] = np.nan
    assert V.validate(row, schema, batch=False) == []          # ค่าว่างรายตัวให้ imputer เติม
    row["Age"] = 10
    assert V.validate(row, schema, batch=False) != []


def test_pipeline_handles_missing_and_unseen_values(loans):
    model = F.build_pipeline(LogisticRegression(max_iter=1000)).fit(loans, F.encode_target(loans[F.TARGET]))
    row = loans.iloc[[0]].copy()
    row["Annual_Income"] = np.nan
    row["Region"] = "Mars"                                        # หมวดใหม่ → one-hot เป็น 0 ไม่ error
    row["Loan_Amount"] = 1e12                                     # outlier → ถูก clip
    p = model.predict_proba(row)[0, 1]
    assert 0.0 <= p <= 1.0


def test_leaky_columns_are_not_features():
    used = set(F.NUMERIC + F.CATEGORICAL)
    for col in F.POST_DECISION_COLS + F.SENSITIVE_COLS + F.ID_COLS + F.EXTERNAL_SCORE_COLS:
        assert col not in used, f"{col} ห้ามใช้เป็นฟีเจอร์"


def test_clip_bounds_learned_from_train_only(loans):
    fe = F.LoanFeatureEngineer().fit(loans)
    lo, hi = fe.bounds_["Loan_Amount"]
    out = fe.transform(loans.assign(Loan_Amount=1e12))
    assert out["Loan_Amount"].max() == hi
