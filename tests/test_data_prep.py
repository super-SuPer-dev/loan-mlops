"""ทดสอบข้อมูลจำลองที่ใช้สาธิต — ถ้าตัวจำลองผิด การสาธิต drift ทั้งหมดจะผิดตาม"""
from src import validation as V
from src.data_prep import concept_drift_policy, make_bad_batch
from src.features import TARGET


def test_concept_drift_changes_only_labels_of_long_loans(loans):
    new = concept_drift_policy(loans)
    feature_cols = [c for c in loans.columns if c != TARGET]
    assert new[feature_cols].equals(loans[feature_cols])           # ข้อมูลเข้าไม่เปลี่ยน
    long_loans = loans["Loan_Term_Months"] > 120
    assert (new.loc[long_loans, TARGET] == "Rejected").all()       # กู้เกิน 120 เดือน = ปฏิเสธ
    assert new.loc[~long_loans, TARGET].equals(loans.loc[~long_loans, TARGET])
    assert (loans.loc[long_loans, TARGET] == "Approved").any()     # มีบางแถวที่ถูกเปลี่ยนจริง


def test_concept_drift_does_not_modify_input(loans):
    before = loans.copy()
    concept_drift_policy(loans)
    assert loans.equals(before)


def test_bad_batch_is_caught_by_schema(loans):
    schema = V.build_schema(loans, target=TARGET)
    problems = V.validate(make_bad_batch(loans), schema)
    assert len(problems) >= 5, problems
