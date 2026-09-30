"""ของที่เทสต์หลายไฟล์ใช้ร่วมกัน — สร้างข้อมูลสังเคราะห์ขนาดเล็กหน้าตาเหมือนชุดจริง ไม่ต้องดาวน์โหลดจาก Kaggle"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import features as F  # noqa: E402


def make_loans(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    income = rng.uniform(20000, 200000, n).round(2)
    df = pd.DataFrame({
        "Application_ID": np.arange(500001, 500001 + n),
        "Age": rng.integers(18, 70, n), "Gender": rng.choice(["Male", "Female"], n),
        "Marital_Status": rng.choice(["Single", "Married"], n),
        "Education_Level": rng.choice(["Bachelor", "Master", "High School"], n),
        "Employment_Status": rng.choice(["Salaried", "Self-Employed", "Unemployed"], n),
        "Job_Experience_Years": rng.integers(0, 40, n), "Annual_Income": income,
        "Monthly_Income": (income / 12).round(2), "Credit_Score": rng.integers(300, 851, n),
        "Credit_History_Years": rng.uniform(0, 30, n).round(1), "Existing_Loans": rng.integers(0, 6, n),
        "Late_Payments_Last_2_Years": rng.integers(0, 5, n), "Loan_Amount": rng.uniform(1000, 400000, n).round(2),
        "Loan_Term_Months": rng.choice([12, 24, 60, 120, 240], n), "Loan_Purpose": rng.choice(["Car", "Home", "Personal"], n),
        "Interest_Rate": rng.uniform(5, 25, n).round(2), "Debt_To_Income_Ratio": rng.uniform(0, 60, n).round(2),
        "Savings_Balance": rng.uniform(0, 100000, n).round(2), "Checking_Balance": rng.uniform(0, 40000, n).round(2),
        "Total_Assets": rng.uniform(5000, 900000, n).round(2), "Total_Liabilities": rng.uniform(100, 300000, n).round(2),
        "Monthly_Debt_Payment": rng.uniform(1, 5000, n).round(2),
        "Property_Ownership": rng.choice(["Rent", "Own", "Mortgage"], n),
        "Residential_Status": rng.choice(["Urban", "Rural"], n), "Dependents": rng.integers(0, 5, n),
        "Co_Applicant": rng.choice(["Yes", "No"], n), "Collateral": rng.choice(["Yes", "No"], n),
        "Application_Channel": rng.choice(["Online", "Branch"], n), "Region": rng.choice(["North", "South"], n),
        "Bank_Account_Age_Years": rng.uniform(0, 30, n).round(1),
        "Previous_Loan_Status": rng.choice(["Paid Off", "Defaulted", "No Previous Loan"], n),
        "Risk_Score": rng.uniform(0, 100, n).round(2), "Default_Risk": rng.choice(["Low", "High"], n),
    })
    # label มีสัญญาณจริงจาก credit score และประวัติผิดนัด ให้โมเดลเรียนได้
    score = (df.Credit_Score - 575) / 80 - 2 * (df.Previous_Loan_Status == "Defaulted") + rng.normal(0, 0.7, n)
    df[F.TARGET] = np.where(score > 0, "Approved", "Rejected")
    df["Approved_Amount"] = np.where(score > 0, df.Loan_Amount * 0.8, 0.0)
    return df


@pytest.fixture(scope="session")
def loans() -> pd.DataFrame:
    return make_loans()
