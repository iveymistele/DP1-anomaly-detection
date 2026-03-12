# app.py
import io
import json
import os
import boto3
import pandas as pd
import requests
import logging
import sys
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks, Request
from baseline import BaselineManager
from processor import process_file

app = FastAPI(title="Anomaly Detection Pipeline")

logger = logging.getLogger("anomaly_app")
logger.setLevel(logging.INFO)
logger.handlers.clear()

formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

file_handler = logging.FileHandler("app.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

s3 = boto3.client("s3")
BUCKET_NAME = os.environ["BUCKET_NAME"]

# ── SNS subscription confirmation + message handler ──────────────────────────

@app.post("/notify")
async def handle_sns(request: Request, background_tasks: BackgroundTasks):
    try:
        body = await request.json()
        msg_type = request.headers.get("x-amz-sns-message-type")
        logger.info(f"Received /notify request with SNS message type: {msg_type}")

        if msg_type == "SubscriptionConfirmation":
            confirm_url = body["SubscribeURL"]
            requests.get(confirm_url, timeout=10)
            logger.info("SNS subscription confirmed successfully")
            return {"status": "confirmed"}

        if msg_type == "Notification":
            s3_event = json.loads(body["Message"])
            record_count = len(s3_event.get("Records", []))
            logger.info(f"SNS notification received with {record_count} record(s)")

            for record in s3_event.get("Records", []):
                key = record["s3"]["object"]["key"]
                logger.info(f"Arrival of new file event received for key: {key}")

                if key.startswith("raw/") and key.endswith(".csv"):
                    logger.info(f"Queueing processing task for file: {key}")
                    background_tasks.add_task(process_file, BUCKET_NAME, key)
                else:
                    logger.info(f"Skipping non-matching object key: {key}")

        return {"status": "ok"}

    except Exception as e:
        logger.exception(f"Error handling SNS request: {e}")
        return {"status": "error", "message": str(e)}


# ── Query endpoints ───────────────────────────────────────────────────────────

@app.get("/anomalies/recent")
def get_recent_anomalies(limit: int = 50):
    try:
        logger.info(f"Request received for /anomalies/recent with limit={limit}")

        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix="processed/")

        keys = sorted(
            [
                obj["Key"]
                for page in pages
                for obj in page.get("Contents", [])
                if obj["Key"].endswith(".csv")
            ],
            reverse=True,
        )[:10]

        logger.info(f"Found {len(keys)} recent processed file(s) to inspect for anomalies")

        all_anomalies = []
        for key in keys:
            response = s3.get_object(Bucket=BUCKET_NAME, Key=key)
            df = pd.read_csv(io.BytesIO(response["Body"].read()))
            if "anomaly" in df.columns:
                flagged = df[df["anomaly"] == True].copy()
                flagged["source_file"] = key
                all_anomalies.append(flagged)

        if not all_anomalies:
            logger.info("No anomalies found across recent processed files")
            return {"count": 0, "anomalies": []}

        combined = pd.concat(all_anomalies).head(limit)
        logger.info(f"Returning {len(combined)} recent anomaly record(s)")
        return {"count": len(combined), "anomalies": combined.to_dict(orient="records")}

    except Exception as e:
        logger.exception(f"Error in /anomalies/recent: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/anomalies/summary")
def get_anomaly_summary():
    try:
        logger.info("Request received for /anomalies/summary")

        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix="processed/")

        summaries = []
        for page in pages:
            for obj in page.get("Contents", []):
                if obj["Key"].endswith("_summary.json"):
                    response = s3.get_object(Bucket=BUCKET_NAME, Key=obj["Key"])
                    summaries.append(json.loads(response["Body"].read()))

        if not summaries:
            logger.info("No processed summary files found yet")
            return {"message": "No processed files yet."}

        total_rows = sum(s["total_rows"] for s in summaries)
        total_anomalies = sum(s["anomaly_count"] for s in summaries)

        logger.info(
            f"Summary calculated across {len(summaries)} file(s): "
            f"{total_anomalies} anomalies out of {total_rows} rows"
        )

        return {
            "files_processed": len(summaries),
            "total_rows_scored": total_rows,
            "total_anomalies": total_anomalies,
            "overall_anomaly_rate": round(total_anomalies / total_rows, 4) if total_rows > 0 else 0,
            "most_recent": sorted(summaries, key=lambda x: x["processed_at"], reverse=True)[:5],
        }

    except Exception as e:
        logger.exception(f"Error in /anomalies/summary: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/baseline/current")
def get_current_baseline():
    try:
        logger.info("Request received for /baseline/current")

        baseline_mgr = BaselineManager(bucket=BUCKET_NAME)
        baseline = baseline_mgr.load()

        channels = {}
        for channel, stats in baseline.items():
            if channel == "last_updated":
                continue
            channels[channel] = {
                "observations": stats["count"],
                "mean": round(stats["mean"], 4),
                "std": round(stats.get("std", 0.0), 4),
                "baseline_mature": stats["count"] >= 30,
            }

        logger.info(f"Baseline current view returned for {len(channels)} channel(s)")

        return {
            "last_updated": baseline.get("last_updated"),
            "channels": channels,
        }

    except Exception as e:
        logger.exception(f"Error in /baseline/current: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/health")
def health():
    try:
        logger.info("Health check called")
        return {"status": "ok", "bucket": BUCKET_NAME, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.exception(f"Error in /health: {e}")
        return {"status": "error", "message": str(e)}