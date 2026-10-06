"""ตรรกะของแต่ละขั้นใน training pipeline — ฟังก์ชัน Python ธรรมดา ไม่ import airflow (แบบ steps.py ของ Lab 11)

ข้อดี: ทดสอบด้วย pytest ได้โดยไม่ต้องยก Airflow, และไฟล์ DAG เหลือแค่ "ลำดับงาน"
ค่าที่ส่งต่อระหว่าง task เป็น dict เล็ก ๆ (path, ตัวเลข) ไม่ใช่ DataFrame เพราะเก็บใน XCom
"""
import hashlib
import json
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src import config as C
from src import features as F
from src import validation as V

# ตัวเลือกโมเดลที่ DAG เทรนขนานกัน (ผลจาก notebook: LogReg ดีพอ ๆ กับ boosting แต่เร็วและอธิบายได้กว่า)
CANDIDATES = {
    "logreg": (LogisticRegression, {"max_iter": 2000}),
    "logreg_c0.1": (LogisticRegression, {"C": 0.1, "max_iter": 2000}),
    "hist_gboost": (HistGradientBoostingClassifier, {"max_iter": 300, "learning_rate": 0.05, "random_state": C.SEED}),
}


# ------------------------------------------------------------------ เครื่องมือช่วย
def file_md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    """อ่านเลข commit จากโฟลเดอร์ .git โดยตรง (ใน container ไม่มีคำสั่ง git)"""
    head = C.ROOT / ".git" / "HEAD"
    if not head.exists():
        return "no-git"
    ref = head.read_text().strip()
    if not ref.startswith("ref:"):
        return ref                                   # detached HEAD = เลข commit อยู่แล้ว
    ref_path = C.ROOT / ".git" / ref.split(" ", 1)[1]
    if ref_path.exists():
        return ref_path.read_text().strip()
    packed = C.ROOT / ".git" / "packed-refs"
    for line in packed.read_text().splitlines() if packed.exists() else []:
        if line.endswith(ref.split(" ", 1)[1]):
            return line.split(" ")[0]
    return "unknown"


def environment_info() -> dict:
    return {"python": platform.python_version(), "os": platform.platform(),
            **{lib: version(lib) for lib in ["scikit-learn", "pandas", "numpy", "mlflow-skinny", "scipy"]}}


def evaluate(model, df: pd.DataFrame, threshold: float) -> dict:
    y = F.encode_target(df[F.TARGET])
    p = model.predict_proba(df)[:, 1]
    pred = (p >= threshold).astype(int)
    return {"roc_auc": float(roc_auc_score(y, p)), "f1": float(f1_score(y, pred)),
            "accuracy": float(accuracy_score(y, pred)), "recall_rejected": float(recall_score(1 - y, 1 - pred))}


def choose_threshold(model, val: pd.DataFrame) -> float:
    """threshold ที่ต้นทุนรวมบน validation ต่ำสุด (อนุมัติผิด 3 : ปฏิเสธผิด 1)"""
    y = F.encode_target(val[F.TARGET]).to_numpy()
    p = model.predict_proba(val)[:, 1]
    ths = np.round(np.arange(0.05, 0.96, 0.01), 2)
    costs = [C.COST_FALSE_APPROVE * ((p >= t) & (y == 0)).sum() + C.COST_FALSE_REJECT * ((p < t) & (y == 1)).sum()
             for t in ths]
    return float(ths[int(np.argmin(costs))])


def latency_p95_ms(model, df: pd.DataFrame, n: int = 200) -> float:
    rows = [df.iloc[[i % len(df)]] for i in range(n)]
    model.predict_proba(rows[0])                     # warm-up
    t = []
    for r in rows:
        t0 = time.perf_counter()
        model.predict_proba(r)
        t.append((time.perf_counter() - t0) * 1000)
    return float(np.percentile(t, 95))


def latency_under_load_p95_ms(model, df: pd.DataFrame, rps: float | None = None, seconds: float | None = None) -> float:
    """p95 ของเวลาตอบ เมื่อคำขอเข้ามาตามนาฬิกาที่อัตรา rps และถูกประมวลผลพร้อมกันหลาย thread (แบบ Flask threaded)

    เวลาของแต่ละคำขอนับจาก "เวลาที่ควรเข้ามา" จึงรวมเวลารอคิวด้วย — โมเดลที่ช้าลงเมื่อถูกเรียกพร้อมกัน
    หรือประมวลผลไม่ทันอัตราเป้าหมาย จะเห็นค่าพุ่งทันที ต่างจาก latency_p95_ms ที่เรียกทีละคำขอ
    """
    rps = rps or C.SLO_TARGET_RPS
    n = max(20, int(rps * (seconds or C.LOAD_CHECK_SECONDS)))
    rows = [df.iloc[[i % len(df)]] for i in range(n)]
    model.predict_proba(rows[0])                     # warm-up

    def call(row, scheduled):
        model.predict_proba(row)
        return (time.perf_counter() - scheduled) * 1000

    t0, futures = time.perf_counter(), []
    with ThreadPoolExecutor(32) as pool:
        for i, row in enumerate(rows):
            scheduled = t0 + i / rps
            delay = scheduled - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            futures.append(pool.submit(call, row, scheduled))
        latencies = [f.result() for f in futures]
    return float(np.percentile(latencies, 95))


def setup_mlflow() -> MlflowClient:
    mlflow.set_tracking_uri(C.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(C.EXPERIMENT)
    return MlflowClient()


def load_champion():
    """โหลดโมเดล @champion จาก registry — ยังไม่มีคืน (None, None)"""
    client = setup_mlflow()
    try:
        mv = client.get_model_version_by_alias(C.MODEL_NAME, C.CHAMPION)
    except MlflowException:
        return None, None
    return mlflow.sklearn.load_model(f"models:/{C.MODEL_NAME}@{C.CHAMPION}"), mv


# ------------------------------------------------------------------ 1. ExampleGen
def example_gen(run_id: str, data_file: str = "historical.csv") -> dict:
    """อ่านข้อมูล แบ่ง train/val/test = 70/15/15 (stratified, seed คงที่) แล้วเขียนเป็นไฟล์ของ run นี้"""
    d = C.run_dir(run_id)
    path = C.DATA / data_file
    if not path.exists() and data_file == C.HISTORICAL_CSV.name:
        # รันครั้งแรกบนเครื่องเปล่า: ดาวน์โหลด + เตรียมข้อมูลให้เอง
        from src.data_prep import prepare_all
        print("เตรียมข้อมูลครั้งแรก:", prepare_all())
    df = pd.read_csv(path)
    train, rest = train_test_split(df, test_size=0.30, random_state=C.SEED, stratify=df[F.TARGET])
    val, test = train_test_split(rest, test_size=0.50, random_state=C.SEED, stratify=rest[F.TARGET])
    out = {"source": str(path), "data_md5": file_md5(path), "run_dir": str(d)}
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(d / f"{name}.csv", index=False, lineterminator="\n")
        out[name] = str(d / f"{name}.csv")
        out[f"n_{name}"] = len(part)
    return out


# ------------------------------------------------------------------ 2. SchemaGen
def schema_gen(examples: dict) -> str:
    """ใช้ schema กลางถ้ามีแล้ว (ห้ามสร้างใหม่ทุกรอบ ไม่งั้นข้อมูลเสียกลายเป็นมาตรฐานใหม่) ไม่มีค่อยสร้างจาก train"""
    if not C.SCHEMA_PATH.exists():
        C.SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
        V.save_schema(V.build_schema(pd.read_csv(examples["train"]), target=F.TARGET), C.SCHEMA_PATH)
    return str(C.SCHEMA_PATH)


# ------------------------------------------------------------------ 3. ExampleValidator
def example_validator(examples: dict, schema_path: str) -> dict:
    schema = V.load_schema(schema_path)
    anomalies = []
    for split in ("train", "val", "test"):
        anomalies += [f"[{split}] {a}" for a in V.validate(pd.read_csv(examples[split]), schema)]
    return {"ok": not anomalies, "anomalies": anomalies}


# ------------------------------------------------------------------ 4. Trainer (รันขนานหนึ่ง task ต่อหนึ่งโมเดล)
def trainer(candidate: str, examples: dict, schema_path: str, airflow_run_id: str = "manual") -> dict:
    """เทรน 1 โมเดล เลือก threshold บน val แล้วบันทึกครบ 6 อย่างลง MLflow"""
    setup_mlflow()
    cls, params = CANDIDATES[candidate]
    train, val = pd.read_csv(examples["train"]), pd.read_csv(examples["val"])
    with mlflow.start_run(run_name=f"{candidate} | {airflow_run_id}") as run:
        model = F.build_pipeline(cls(**params))
        t0 = time.perf_counter()
        model.fit(train, F.encode_target(train[F.TARGET]))
        fit_s = time.perf_counter() - t0
        threshold = choose_threshold(model, val)
        m = evaluate(model, val, threshold)
        m["p95_latency_ms"] = latency_p95_ms(model, val)

        mlflow.set_tags({"git_commit": git_commit(), "airflow_run_id": airflow_run_id,          # (1) โค้ด
                         "candidate": candidate})
        mlflow.log_params({"data_source": Path(examples["source"]).name,                        # (2) ข้อมูล
                           "data_md5": examples["data_md5"], "n_train": examples["n_train"]})
        mlflow.log_params({"model_type": cls.__name__, "threshold": threshold,                  # (3) hyperparameter
                           **{f"clf__{k}": v for k, v in params.items()}})
        mlflow.log_metrics({**{f"val_{k}": v for k, v in m.items()}, "fit_seconds": fit_s})    # (4) metric
        info = mlflow.sklearn.log_model(sk_model=model, name="model",                            # (5) artifact
                                        input_example=train[F.NUMERIC + F.CATEGORICAL].head(3),
                                        code_paths=[str(C.ROOT / "src")],
                                        skops_trusted_types=C.TRUSTED_TYPES)
        mlflow.log_artifact(schema_path)              # API ดาวน์โหลด schema นี้ไปตรวจคำขอ
        mlflow.log_dict(environment_info(), "environment.json")                                   # (6) สภาพแวดล้อม
        if (C.ROOT / "uv.lock").exists():
            mlflow.log_artifact(str(C.ROOT / "uv.lock"))
    print(f"{candidate}: val_auc={m['roc_auc']:.4f} threshold={threshold} p95={m['p95_latency_ms']:.1f}ms")
    return {"candidate": candidate, "run_id": run.info.run_id, "model_uri": info.model_uri,
            "threshold": threshold, "val_roc_auc": m["roc_auc"], "p95_latency_ms": m["p95_latency_ms"]}


def select_best(results: list[dict], examples: dict | None = None) -> dict:
    """Satisficing + Optimizing: ตัดตัวที่ช้าเกินงบ latency ทิ้งก่อน แล้วค่อยเลือก val AUC สูงสุดจากที่เหลือ

    latency วัด "ที่โหลดเป้าหมายของ SLO" ทีละโมเดลตามลำดับ (ไม่วัดใน trainer เพราะ trainer รันขนานกัน จะกวนกันเอง)
    ถ้าไม่ส่ง examples มา (เช่นเรียกจากเทสต์) จะใช้ค่าแบบทีละคำขอที่ trainer วัดไว้แทน
    ถ้า AUC ต่างกันไม่ถึง 0.003 เลือกตัวที่อยู่ก่อนใน CANDIDATES (ตัวที่ง่ายและเร็วกว่า)
    ถ้าไม่มีตัวไหนผ่านงบ latency เลย เลือกตัวที่เร็วที่สุดส่งไปให้ evaluator ตัดสิน (ซึ่งจะไม่ผ่านด่าน)
    """
    order = list(CANDIDATES)
    results = sorted(results, key=lambda r: order.index(r["candidate"]))
    if examples is not None:
        setup_mlflow()
        val = pd.read_csv(examples["val"])
        for r in results:
            r["p95_latency_ms"] = latency_under_load_p95_ms(mlflow.sklearn.load_model(r["model_uri"]), val)
    for r in results:
        print(f"  {r['candidate']:12s} val_auc={r['val_roc_auc']:.4f} p95@{C.SLO_TARGET_RPS}rps={r['p95_latency_ms']:.1f}ms"
              f"{'' if r['p95_latency_ms'] <= C.MAX_P95_LATENCY_MS else '  ✗ เกินงบ latency'}")
    eligible = [r for r in results if r["p95_latency_ms"] <= C.MAX_P95_LATENCY_MS]
    if not eligible:
        return min(results, key=lambda r: r["p95_latency_ms"])
    best = eligible[0]
    for r in eligible[1:]:
        if r["val_roc_auc"] > best["val_roc_auc"] + 0.003:
            best = r
    print("→ เลือก", best["candidate"])
    return best


# ------------------------------------------------------------------ 5. Evaluator (ด่านตรวจ)
def evaluator(examples: dict, best: dict) -> dict:
    """วัดผลบน test (ครั้งเดียว) และเทียบกับ champion ปัจจุบันบน test ชุดเดียวกัน"""
    client = setup_mlflow()
    test = pd.read_csv(examples["test"])
    model = mlflow.sklearn.load_model(best["model_uri"])
    m = evaluate(model, test, best["threshold"])
    m["p95_latency_ms"] = latency_under_load_p95_ms(model, test)     # วัดที่โหลดเป้าหมายของ SLO
    size_path = Path(examples["run_dir"]) / "size_check.joblib"
    joblib.dump(model, size_path)
    m["model_size_mb"] = size_path.stat().st_size / 1e6

    champion, champ_mv = load_champion()
    if champion is None:
        m["champion_version"], m["champion_roc_auc"], m["improvement"] = None, None, None
    else:
        champ_auc = evaluate(champion, test, float(champ_mv.tags.get("threshold", 0.5)))["roc_auc"]
        m["champion_version"], m["champion_roc_auc"] = champ_mv.version, champ_auc
        m["improvement"] = m["roc_auc"] - champ_auc

    checks = {
        "roc_auc": m["roc_auc"] >= C.MIN_ROC_AUC,
        "recall_rejected": m["recall_rejected"] >= C.MIN_RECALL_REJECTED,
        "p95_latency": m["p95_latency_ms"] <= C.MAX_P95_LATENCY_MS,
        "model_size": m["model_size_mb"] <= C.MAX_MODEL_SIZE_MB,
        "vs_champion": m["improvement"] is None or m["improvement"] >= C.MIN_IMPROVEMENT,
    }
    m["checks"] = checks
    m["blessed"] = all(checks.values())
    client.log_metric(best["run_id"], "test_roc_auc", m["roc_auc"])
    client.log_metric(best["run_id"], "test_recall_rejected", m["recall_rejected"])
    client.log_metric(best["run_id"], "test_p95_latency_ms", m["p95_latency_ms"])
    client.set_tag(best["run_id"], "blessed", str(m["blessed"]))
    Path(examples["run_dir"], "evaluation.json").write_text(json.dumps(m, indent=2, default=str))
    return m


def fairness_audit(examples: dict, best: dict) -> dict:
    """อัตราอนุมัติที่โมเดลทายแยกตามเพศ (Gender ไม่ได้เป็นฟีเจอร์ แต่อาจแฝงผ่านฟีเจอร์อื่น)"""
    setup_mlflow()
    test = pd.read_csv(examples["test"])
    model = mlflow.sklearn.load_model(best["model_uri"])
    test["approve"] = model.predict_proba(test)[:, 1] >= best["threshold"]
    rates = test.groupby("Gender")["approve"].mean().round(4).to_dict()
    gap = max(rates.values()) - min(rates.values())
    result = {"approval_rate_by_gender": rates, "gap": round(gap, 4), "ok": gap <= C.MAX_APPROVAL_GAP}
    MlflowClient().log_metric(best["run_id"], "fairness_approval_gap", gap)
    return result


# ------------------------------------------------------------------ 6. Registry
def register(best: dict, metrics: dict, promote: bool, airflow_run_id: str = "manual") -> dict:
    """ลงทะเบียนทุกโมเดลที่ผ่านการประเมิน พร้อมสถานะ

    promote=True  → ย้าย @champion มาที่เวอร์ชันนี้ และให้ @previous ชี้ champion เดิม (ไว้ rollback)
    promote=False → ติด tag status=rejected เก็บไว้เป็นหลักฐานว่าเคยมีตัวที่ไม่ผ่านด่าน
    """
    client = setup_mlflow()
    mv = mlflow.register_model(best["model_uri"], C.MODEL_NAME)
    tags = {"threshold": best["threshold"], "candidate": best["candidate"], "test_roc_auc": round(metrics["roc_auc"], 4),
            "airflow_run_id": airflow_run_id, "status": "approved" if promote else "rejected",
            "registered_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    for k, v in tags.items():
        client.set_model_version_tag(C.MODEL_NAME, mv.version, k, str(v))
    if promote:
        if metrics.get("champion_version"):
            client.set_registered_model_alias(C.MODEL_NAME, "previous", metrics["champion_version"])
        client.set_registered_model_alias(C.MODEL_NAME, C.CHAMPION, mv.version)
    else:
        client.set_registered_model_alias(C.MODEL_NAME, "challenger", mv.version)
    return {"version": mv.version, **tags}


def rollback(to_version: str | None = None) -> dict:
    """ย้าย @champion กลับไปที่เวอร์ชันที่ระบุ (ไม่ระบุ = ไปที่ @previous)"""
    client = setup_mlflow()
    current = client.get_model_version_by_alias(C.MODEL_NAME, C.CHAMPION).version
    target = to_version or client.get_model_version_by_alias(C.MODEL_NAME, "previous").version
    client.set_registered_model_alias(C.MODEL_NAME, C.CHAMPION, target)
    client.set_registered_model_alias(C.MODEL_NAME, "previous", current)
    client.set_model_version_tag(C.MODEL_NAME, current, "status", "rolled_back")
    return {"from": current, "to": target}
