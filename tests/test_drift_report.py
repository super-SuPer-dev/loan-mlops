"""ทดสอบรายงาน Evidently: ข้อมูลที่ไม่เปลี่ยนต้อง drift น้อย ข้อมูลที่เลื่อนการกระจายต้อง drift มาก"""
import pytest

pytest.importorskip("evidently")

from conftest import make_loans  # noqa: E402

from src.drift_report import build_drift_report  # noqa: E402


def test_same_distribution_has_little_drift(tmp_path):
    ref, cur = make_loans(600, seed=0), make_loans(600, seed=1)   # สุ่มจากการกระจายเดียวกัน
    out = build_drift_report(ref, cur, tmp_path / "same.html")
    assert (tmp_path / "same.html").stat().st_size > 10_000
    assert out["drift_share"] < 0.2


def test_shifted_distribution_is_detected(tmp_path):
    ref, cur = make_loans(600, seed=0), make_loans(600, seed=1)
    cur["Credit_Score"] = (cur["Credit_Score"] - 200).clip(lower=300)    # เศรษฐกิจถดถอย
    cur["Annual_Income"] = cur["Annual_Income"] * 0.5
    cur["Savings_Balance"] = cur["Savings_Balance"] * 0.3
    out = build_drift_report(ref, cur, tmp_path / "shifted.html")
    assert out["drifted_columns"] >= 3
    assert out["n_columns"] == 27
