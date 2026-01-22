import os
import json
import uuid
from datetime import datetime, date
from typing import Optional, List, Dict

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from google.cloud import storage

from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset, DataSummaryPreset


app = FastAPI(title="Language Drift Monitoring API")

BUCKET_NAME = os.getenv("BUCKET_NAME", "")
LOG_PREFIX = os.getenv("LOG_PREFIX", "inference_logs")
PREDICTION_SERVICE = os.getenv("PREDICTION_SERVICE", "language-detection-api")
REFERENCE_PATH = os.getenv("REFERENCE_PATH", "data/raw/language_detection.csv")
REPORT_PREFIX = os.getenv("REPORT_PREFIX", "drift_reports")

gcs = storage.Client()


def load_reference_df(max_rows: int = 5000) -> pd.DataFrame:
    if not os.path.exists(REFERENCE_PATH):
        raise FileNotFoundError(f"REFERENCE_PATH not found: {REFERENCE_PATH}")

    df = pd.read_csv(REFERENCE_PATH)
    # expected columns in your dataset: Text, Language
    if "Text" not in df.columns or "Language" not in df.columns:
        raise ValueError(f"Reference CSV must contain columns Text, Language. Found: {df.columns.tolist()}")

    df = df.rename(columns={"Text": "content", "Language": "target"})
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42)
    return df.reset_index(drop=True)


def list_log_blobs(day_str: str) -> List[storage.Blob]:
    prefix = f"{LOG_PREFIX}/service={PREDICTION_SERVICE}/day={day_str}/"
    bucket = gcs.bucket(BUCKET_NAME)
    blobs = list(bucket.list_blobs(prefix=prefix))
    # blob name includes timestamp -> lexicographic sort works for “latest”
    blobs.sort(key=lambda b: b.name)
    return blobs


def load_current_df(n: int, day_str: str) -> pd.DataFrame:
    blobs = list_log_blobs(day_str)
    if not blobs:
        return pd.DataFrame(columns=["content", "target"])

    blobs = blobs[-n:]
    rows: List[Dict[str, str]] = []
    for b in blobs:
        payload = json.loads(b.download_as_text())
        rows.append(
            {
                "content": payload.get("input_text", ""),
                "target": payload.get("predicted_language", ""),
            }
        )
    return pd.DataFrame(rows)


def build_eval(current_df: pd.DataFrame, reference_df: pd.DataFrame):
    schema = DataDefinition(
        text_columns=["content"],
        categorical_columns=["target"],
    )

    current_ds = Dataset.from_pandas(current_df, data_definition=schema)
    reference_ds = Dataset.from_pandas(reference_df, data_definition=schema)

    report = Report([DataSummaryPreset(), DataDriftPreset()])
    # first arg = current, second = reference
    my_eval = report.run(current_ds, reference_ds)
    return my_eval


@app.get("/health")
def health():
    if not BUCKET_NAME:
        return JSONResponse({"status": "warning", "msg": "BUCKET_NAME is empty"})
    return {"status": "ok"}


@app.get("/report", response_class=HTMLResponse)
def report(
    n: int = Query(200, ge=10, le=5000),
    day: Optional[str] = None,
    upload: bool = True,
):
    """
    Generates a drift report by comparing:
      - current data: last N inference logs from GCS
      - reference data: local REFERENCE_PATH CSV

    Returns HTML.
    """
    if not BUCKET_NAME:
        raise HTTPException(status_code=500, detail="BUCKET_NAME env var is not set")

    day_str = day or date.today().isoformat()

    reference_df = load_reference_df()
    current_df = load_current_df(n=n, day_str=day_str)

    if current_df.empty:
        raise HTTPException(
            status_code=400,
            detail=f"No inference logs found for service={PREDICTION_SERVICE} day={day_str}",
        )

    my_eval = build_eval(current_df, reference_df)

    out_path = "/tmp/drift_report.html"
    my_eval.save_html(out_path)

    with open(out_path, "r", encoding="utf-8") as f:
        html = f.read()

    if upload:
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        name = (
            f"{REPORT_PREFIX}/service={PREDICTION_SERVICE}/day={day_str}/"
            f"{ts}_{uuid.uuid4().hex}.html"
        )
        bucket = gcs.bucket(BUCKET_NAME)
        bucket.blob(name).upload_from_string(html, content_type="text/html")

    return HTMLResponse(content=html)
