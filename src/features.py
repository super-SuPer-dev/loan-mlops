"""การแปลงข้อมูลของโปรเจกต์ — ไฟล์เดียวที่ทั้งตอนเทรนและตอนให้บริการ import ไปใช้

ทำไมต้องแยกเป็นไฟล์นี้ (กัน Training-Serving Skew แบบใน Lab 7 / Lab 11)
  ถ้าตอนเทรนเขียนการแปลงไว้ใน notebook แล้วตอนให้บริการเขียนใหม่ใน API
  สองที่นี้จะเพี้ยนกันสักวัน เช่น ลืมคำนวณฟีเจอร์ตัวหนึ่ง หรือใช้ค่า median คนละค่า
  โมเดลจะได้ข้อมูลหน้าตาไม่เหมือนตอนเทรน และทำนายผิดโดยไม่มี error ให้เห็น
  เราจึงเอาการแปลงทั้งหมดใส่ไว้ใน sklearn Pipeline ก้อนเดียวกับโมเดล แล้วบันทึกเป็น artifact เดียว
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Loan_Approved"
POSITIVE_LABEL = "Approved"  # คลาส 1 = อนุมัติ, คลาส 0 = ปฏิเสธ

# ---- คอลัมน์ที่ "ห้าม" ใช้เป็นฟีเจอร์ แยกตามเหตุผล ----
ID_COLS = ["Application_ID"]  # เลขรันเฉย ๆ ไม่มีความหมาย
# รู้ค่าได้ "หลัง" ธนาคารตัดสินใจแล้วเท่านั้น → ถ้าใช้คือโกงข้อสอบ (data leakage)
POST_DECISION_COLS = ["Approved_Amount", "Default_Risk", "Interest_Rate"]
# คะแนนที่ระบบเดิมคำนวณมาให้ ไม่รู้สูตร และสัมพันธ์กับผลอนุมัติสูงมาก (ดูการทดลอง ablation ใน notebook)
EXTERNAL_SCORE_COLS = ["Risk_Score"]
# ข้อมูลอ่อนไหว ไม่ใช้ตัดสินสินเชื่อ แต่เก็บไว้ตรวจความเป็นธรรม (fairness) ภายหลัง
SENSITIVE_COLS = ["Gender", "Marital_Status"]
# ซ้ำกับ Annual_Income / 12 แบบเป๊ะ ๆ ใช้ตรวจความสอดคล้องของข้อมูลอย่างเดียว
REDUNDANT_COLS = ["Monthly_Income"]

NUMERIC = [
    "Age", "Job_Experience_Years", "Annual_Income", "Credit_Score", "Credit_History_Years",
    "Existing_Loans", "Late_Payments_Last_2_Years", "Loan_Amount", "Loan_Term_Months",
    "Debt_To_Income_Ratio", "Savings_Balance", "Checking_Balance", "Total_Assets",
    "Total_Liabilities", "Monthly_Debt_Payment", "Dependents", "Bank_Account_Age_Years",
]
CATEGORICAL = [
    "Education_Level", "Employment_Status", "Loan_Purpose", "Property_Ownership",
    "Residential_Status", "Co_Applicant", "Collateral", "Application_Channel", "Region",
    "Previous_Loan_Status",
]
# ฟีเจอร์ที่เราสร้างเพิ่มเอง (คำนวณใน LoanFeatureEngineer)
ENGINEERED = ["Loan_To_Income", "Liability_To_Asset"]


def encode_target(y: pd.Series) -> pd.Series:
    """แปลง "Approved"/"Rejected" เป็น 1/0"""
    return (y == POSITIVE_LABEL).astype(int)


class LoanFeatureEngineer(BaseEstimator, TransformerMixin):
    """ขั้นแรกของ Pipeline: สร้างฟีเจอร์อัตราส่วน แล้วตัดค่าผิดปกติ (outlier) ด้วยขอบที่เรียนจากชุด train

    - fit()      : จำขอบล่าง/บน (quantile 1% และ 99%) ของทุกคอลัมน์ตัวเลข จากชุด train เท่านั้น
    - transform(): ใช้ขอบที่จำไว้ตัดค่าที่ล้นออกไป (clip) ทั้งตอนเทรนและตอนให้บริการ

    ทำไมต้อง clip แทนการลบแถวทิ้ง: ตอนให้บริการเราลบคำขอของลูกค้าทิ้งไม่ได้ ต้องตอบทุกคำขอ
    ค่าว่าง (NaN) ปล่อยผ่านไปก่อน ให้ SimpleImputer ในขั้นถัดไปเติม
    """

    def __init__(self, numeric_cols=None, lower_q=0.01, upper_q=0.99):
        self.numeric_cols = numeric_cols
        self.lower_q = lower_q
        self.upper_q = upper_q

    @staticmethod
    def _add_ratios(X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # หารด้วย 0 จะได้ inf → เปลี่ยนเป็น NaN ให้ imputer จัดการ
        X["Loan_To_Income"] = (X["Loan_Amount"] / X["Annual_Income"]).replace([np.inf, -np.inf], np.nan)
        X["Liability_To_Asset"] = (X["Total_Liabilities"] / X["Total_Assets"]).replace([np.inf, -np.inf], np.nan)
        return X

    def fit(self, X, y=None):
        X = self._add_ratios(pd.DataFrame(X))
        cols = list(self.numeric_cols or NUMERIC) + ENGINEERED
        self.bounds_ = {c: (float(X[c].quantile(self.lower_q)), float(X[c].quantile(self.upper_q))) for c in cols}
        return self

    def transform(self, X):
        X = self._add_ratios(pd.DataFrame(X))
        cols = list(self.bounds_)
        lo = pd.Series({c: b[0] for c, b in self.bounds_.items()})
        hi = pd.Series({c: b[1] for c, b in self.bounds_.items()})
        # clip ทุกคอลัมน์ในคำสั่งเดียว (เร็วกว่าวนทีละคอลัมน์หลายเท่า สำคัญตอนให้บริการทีละคำขอ)
        X[cols] = X[cols].astype(float).clip(lower=lo, upper=hi, axis=1)
        return X


def build_pipeline(clf, numeric_cols=None, categorical_cols=None) -> Pipeline:
    """ประกอบ Pipeline เต็มก้อน: สร้างฟีเจอร์ → เติมค่าว่าง/ปรับสเกล/one-hot → โมเดล

    ตัว Pipeline ที่ได้คือ "สิ่งเดียว" ที่ถูกบันทึกลง MLflow และถูกโหลดไปใช้ใน API
    """
    numeric_cols = list(numeric_cols or NUMERIC)
    categorical_cols = list(categorical_cols or CATEGORICAL)
    pre = ColumnTransformer(
        [
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),        # ค่าว่างตัวเลข → median ของ train
                ("scale", StandardScaler()),
            ]), numeric_cols + ENGINEERED),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),  # ค่าว่างหมวดหมู่ → ค่าที่พบบ่อยสุด
                ("onehot", OneHotEncoder(handle_unknown="ignore")),   # หมวดใหม่ที่ไม่เคยเห็น → เป็น 0 ทั้งหมด ไม่ error
            ]), categorical_cols),
        ],
        sparse_threshold=0,  # คืนค่าเป็น dense เพื่อให้ใช้ได้กับทุกโมเดล (บทเรียนจาก Lab 11)
    )
    return Pipeline([
        ("features", LoanFeatureEngineer(numeric_cols=numeric_cols)),
        ("pre", pre),
        ("clf", clf),
    ])
